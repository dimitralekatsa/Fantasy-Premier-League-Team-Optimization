import pandas as pd
import requests
import numpy as np
import time
import json
import os
from tqdm import tqdm
import concurrent.futures
from typing import Any
import random

from fpl.utils.utils import (
    get_fpl_players_details,
    convert_season_format,
    get_team_info,
    get_fixture_difficulties,
    append_team_metadata,
)

class FPLPastFixturesFetcher:
    """
    Class for fetching current season data from FPL API
    """
    def __init__(
            self,
            path: str = "",
            save_data: bool = True,
            current_season: str = "", 
            hard_fetch: bool = False,
            player_status: list[str] = ["a", "d", "i", "u"],
            player_details_df: pd.DataFrame | None= None,
    ):
        """        
        Parameters:
            path: Base path for data storage
            save_data: Whether to save processed data to files
            current_season: Current FPL season in 'YY-YY' format
            hard_fetch: If True, fetches all data regardless of existing data
            player_status: Status values to filter players by
            player_details_df: Optional pre-loaded player details dataframe
        """
        self.path = path
        self.save_data = save_data
        self.current_season = current_season
        self.hard_fetch = hard_fetch
        self.player_status = set(player_status)
        self.player_details_df = player_details_df
        
        if path:
            os.makedirs(path, exist_ok=True)
    
    def fetch_fpl_players(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Fetch player and team data from FPL API
        """
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        try:
            # Add a small random delay to avoid potential rate limits
            time.sleep(random.uniform(0.1, 0.5))
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            players = data["elements"]
            players = [player for player in players if player["status"] in self.player_status]
            teams = data["teams"]
            return players, teams
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return [], []
    
    def fetch_player_history(self, player_id: int) -> list[dict[str, Any]]:
        """
        Fetch historical data for a specific player
        """
        url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        try:
            time.sleep(random.uniform(0.1, 0.5))
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "history" not in data:
                print(f"Warning: 'history' not found in data for player {player_id}")
                return []
            return data["history"]
        except requests.exceptions.RequestException as e:
            retry_delay = random.uniform(1.0, 3.0)
            print(f"Failed to fetch history for player {player_id}: {e}. Retrying in {retry_delay:.2f} seconds...")
            time.sleep(retry_delay)
            
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                if "history" not in data:
                    print(f"Warning: 'history' not found in data for player {player_id}")
                    return []
                return data["history"]
            except requests.exceptions.RequestException as retry_e:
                print(f"Retry also failed for player {player_id}: {retry_e}")
                return []
    
    def fetch_fixtures(self) -> list[dict[str, Any]]:
        """
        Fetch all fixtures data from FPL API
        """
        url = "https://fantasy.premierleague.com/api/fixtures/"
        try:
            time.sleep(random.uniform(0.1, 0.5))
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch fixtures: {e}")
            return []

    def fetch_incremental_player_history(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch player history data incrementally, only fetching new gameweeks
        Retuens existing data and new data for different processing
        """
        start_time = time.time()
        
        if self.player_details_df is None or self.player_details_df.empty:
            try:
                self.player_details_df = get_fpl_players_details(self.path, save_data=True)
                if self.player_details_df.empty:
                    raise Exception("Function returned empty DataFrame")
            except Exception as e:
                print(f"Could not fetch player details from API: {e}")
                # If fetching fails, try to load from file
                try:
                    self.player_details_df = pd.read_csv(os.path.join(self.path, "player_details.csv"))
                    print("Loaded player details from existing file")
                except Exception as load_e:
                    print(f"Could not load player details from file: {load_e}")
                    print("Warning: Player details not available. Proceeding without them.")
        
        existing_data = None
        max_existing_gw = 0

        folder_season = convert_season_format(self.current_season, target="long_folder")
        file_season = convert_season_format(self.current_season, target="file")
        
        folder_path = os.path.join(self.path, folder_season)
        os.makedirs(folder_path, exist_ok=True)

        if not self.hard_fetch:
            processed_data_path = os.path.join(folder_path, f"{file_season}_past_fixtures.csv")
            final_data_path = os.path.join(folder_path, f"{file_season}_dataset.csv")

            current_season_file = None

            if os.path.exists(final_data_path):
                current_season_file = final_data_path
                print(f"Found final dataset file: (GW38 completed)")
            elif os.path.exists(processed_data_path):
                current_season_file = processed_data_path

            if current_season_file:
                existing_data = pd.read_csv(current_season_file)
                if not existing_data.empty:
                    max_existing_gw = existing_data["gameweek"].max()
                    # Remove data from max_existing_gw to ensure completeness
                    existing_data = existing_data[existing_data["gameweek"] < max_existing_gw]
                    print(f"Loaded existing data up to GW{max_existing_gw-1}. Will fetch GW{max_existing_gw} onwards.")
                else:
                    max_existing_gw = 1
                    print("Current season data file is empty. Will fetch all gameweeks.")
        else:
            print("Hard fetch enabled. Fetching all gameweeks from the start of the season.")
            max_existing_gw = 1
        
        players, teams = self.fetch_fpl_players()
        fixtures = self.fetch_fixtures()
        
        if fixtures:
            fixtures_df = pd.DataFrame(fixtures)
            fixtures_df.to_csv(os.path.join(folder_path, f"{file_season}_fixtures.csv"), index=False)
        
        if not players or not teams or not fixtures:
            print("Failed to fetch essential data. Aborting.")
            return existing_data if existing_data is not None else pd.DataFrame(), pd.DataFrame()
        
        all_data = []
        
        # Limit concurrent requests to avoid overwhelming the API
        max_workers = 5
        
        with tqdm(desc="Fetching players", total=len(players)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_player = {
                    executor.submit(self.process_player, player, teams, fixtures, max_existing_gw): player 
                    for player in players
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_player):
                    player = future_to_player[future]
                    try:
                        player_data = future.result()
                        all_data.extend(player_data)
                    except Exception as e:
                        print(f"Error processing player {player['web_name']}: {e}")
                    finally:
                        pbar.update(1)
        
        if not all_data:
            print("No new data found to add.")
            existing = existing_data if existing_data is not None else pd.DataFrame()
            return existing, pd.DataFrame()
            
        new_df = pd.DataFrame(all_data)

        # Remove redundant columns
        columns_to_remove = ["element"]
        for col in columns_to_remove:
            if col in new_df.columns:
                new_df.drop(columns=col, inplace=True)

        int8_columns = [
            "assists", "bonus", "clean_sheets", "goals_conceded", "goals_scored", "own_goals",
            "penalties_missed", "penalties_saved", "red_cards", "gameweek", "yellow_cards",
            "team_a_score", "team_h_score", "penalties_order", "corners_and_indirect_freekicks_order",
            "direct_freekicks_order", "opponent_difficulty", "team_difficulty", "saves",
            "mng_clean_sheets", "mng_draw", "mng_goals_scored", "mng_loss", "mng_underdog_draw",
            "mng_underdog_win", "mng_win", 
        ]

        int16_columns = [
            "bps", "fixture", "minutes", "threat", "total_points", "value", 
        ]

        int32_columns = [
            "team_strength", "team_strength_overall", "team_strength_attack", "team_strength_defence",
            "opponent_team_id", "player_id", "team_id", 
        ]

        int64_columns = [
            "code", "team_code", "opponent_team_code", 
        ]

        float_columns = [
            "creativity", "expected_assists", "expected_goal_involvements", "expected_goals",
            "expected_goals_conceded", "ict_index", "influence", 
        ]


        float_columns = ["creativity", "expected_assists", "expected_goal_involvements", "expected_goals",
                        "expected_goals_conceded", "ict_index", "influence", "xP", 
        ]

        bool_columns = ["was_home"]

        for col in int8_columns:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0).astype("Int8")

        for col in int16_columns:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0).astype("Int16")

        for col in int32_columns:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0).astype("Int32")

        for col in int64_columns:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0).astype("Int64")

        for col in float_columns:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0.0).astype("float32")
        
        for col in bool_columns:
            if col in new_df.columns:
                # Boolean conversion that handles multiple input takes
                new_df[col] = new_df[col].map({
                    True: True, "True": True, "true": True, 1: True, "1": True,
                    False: False, "False": False, "false": False, 0: False, "0": False,
                    None: False, "None": False, "": False, np.nan: False
                }).fillna(False).astype("bool")
        
        # Skip if no new data
        if new_df.empty:
            print("No new data found to add.")
            existing = existing_data if existing_data is not None else pd.DataFrame()
            return existing, pd.DataFrame()
        
        # Add player metadata if available
        if self.player_details_df is not None and not self.player_details_df.empty:
            try:
                merge_columns = [
                    "code", "position", "corners_and_indirect_freekicks_order",
                    "direct_freekicks_order", "penalties_order", "status", "now_cost"
                ]
                
                # Filter columns that actually exist
                merge_columns = [col for col in merge_columns if col in self.player_details_df.columns]
                
                if merge_columns:
                    new_df = pd.merge(
                        new_df,
                        self.player_details_df[merge_columns],
                        on=["code"],
                        how="left",
                        suffixes=("", "_from_details")
                    )
                    
                    # If players_details has status column, use it to override the status in new_df
                    if "status_from_details" in new_df.columns:
                        new_df["status"] = new_df["status_from_details"].fillna(new_df["status"])
                        new_df.drop("status_from_details", axis=1, inplace=True)
                
                # Fill NAs for order columns
                order_columns = ["corners_and_indirect_freekicks_order", "direct_freekicks_order", "penalties_order"]
                for col in order_columns:
                    if col in new_df.columns:
                        new_df[col] = new_df[col].fillna(5).astype("Int32")
            except Exception as e:
                print(f"Warning: Could not merge player details: {e}")
        
        max_fetched_gw = int(new_df["gameweek"].max()) if not new_df.empty else 0
        metadata = {
            "last_processed_gw": max_fetched_gw,
            "last_fetch_time": time.time(),
            "season": self.current_season,
            "was_hard_fetch": self.hard_fetch,
            "player_status_included": list(self.player_status)
        }
        
        with open(f"{self.path}processing_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        if not self.hard_fetch and existing_data is not None and not existing_data.empty:
            # Update only status to reflect current availability
            status_map = dict(zip(new_df["code"], new_df["status"]))
            existing_data["status"] = existing_data["code"].map(status_map)
            
            # Add a timestamp for when status was last updated
            status_update_time = time.time()
            existing_data["status_updated_at"] = status_update_time
            new_df["status_updated_at"] = status_update_time

            elapsed_time = time.time() - start_time
            print(f"Data fetch completed in {elapsed_time:.2f} seconds.")
            return existing_data, new_df
        else:
            if "status" in new_df.columns:
                new_df["status_updated_at"] = time.time()

            elapsed_time = time.time() - start_time
            print(f"Data fetch completed in {elapsed_time:.2f} seconds.")
            return pd.DataFrame(), new_df

    def process_player(
            self,
            player: dict[str, Any],
            teams: list[dict[str, Any]],
            fixtures: list[dict[str, Any]], 
            min_gameweek: int,
        )-> list[dict[str, Any]]:
        """
        Process an individual player's data
        """
        player_data = []
        player_id = player["id"]
        player_code = player["code"]
        player_name = player["web_name"]
        team_id = player["team"]
        team_name, team_code = get_team_info(team_id, teams)
        player_status = player.get("status", "")
        
        try:
            history = self.fetch_player_history(player_id)
            
            for game in history:
                if game["round"] >= min_gameweek:
                    opponent_team_id = game["opponent_team"]
                    opponent_team_name, opponent_team_code = get_team_info(opponent_team_id, teams)
                    opponent_difficulty, team_difficulty, was_home = get_fixture_difficulties(
                        game["round"], team_id, fixtures
                    )
                    
                    gameweek_data = dict(game)
                    gameweek_data.update({
                        "player_id": player_id,
                        "code": player_code,
                        "player_name": player_name,
                        "team_id": team_id,
                        "team": team_name,
                        "team_code": team_code,
                        "opponent_team_id": opponent_team_id,
                        "opponent_team": opponent_team_name,
                        "opponent_team_code": opponent_team_code,
                        "opponent_difficulty": opponent_difficulty,
                        "team_difficulty": team_difficulty,
                        "season": self.current_season,
                        "was_home": was_home,
                        "status": player_status,
                    })
                    
                    gameweek_data["gameweek"] = gameweek_data.pop("round")
                    player_data.append(gameweek_data)
        except Exception as e:
            print(f"Error processing player {player_name} (ID: {player_id}): {e}")
        
        return player_data


class FPLPastFixturesProcessor:
    """
    Class for processing current season data with team metadata
    """
    def __init__(
            self,
            path: str = "",
            save_data: bool = True,
            current_season: str = "",
            update_team_metadata: bool = False):
        """     
        Parameters:
            path: Path for data storage
            save_data: Whether to save processed data to files
            current_season: current FPL season in 'YY-YY' format
            update_team_metadata: Whether to update team metadata fro existing rows
        """
        self.path = path
        self.save_data = save_data
        self.current_season = current_season
        self.update_team_metadata = update_team_metadata

    def fetch_current_teams_data(self) -> pd.DataFrame:
        """
        Fetch current teams data from FPL API and return raw DataFrame
        """
        try:
            url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            teams_df = pd.DataFrame(data["teams"])
            return teams_df
            
        except Exception as e:
            print(f"Error fetching teams data: {e}")
            return pd.DataFrame()

    def process_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes position fields to long format (Defender, Midfielder, Goalkeeper, Forward)
        """
        position_dict = {
            "DEF": "Defender",
            "FWD": "Forward",
            "GK": "Goalkeeper",
            "MID": "Midfielder",
            "MNG": "Manager",
        }

        # Check if long-format position names exist
        short_formats = any(pos in df["position"].unique() for pos in position_dict.keys())

        if short_formats:
            # Map long-format position names to short-format
            df["position"] = df["position"].map(lambda x: position_dict.get(x, x))

        return df

    def process_current_season_data(
            self,
            existing_df: pd.DataFrame | None = None,
            new_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process current season data with team metadata
        Only processes new data if update_team_metadata is False
        
        Parameters:
            existing_df: DataFrame with already processed season data 
            new_df: DataFrame with new gameweeks that the fetching returned
            
        Returns:
            processed_new: new_df that is processed to contain team_metadata
            combined_df: existing_df + new_df after with added team metadata
        """
        file_season = convert_season_format(self.current_season, target="file")
        long_folder_season = convert_season_format(self.current_season, target="long_folder")
        folder_path = os.path.join(self.path, long_folder_season)
        os.makedirs(folder_path, exist_ok=True)
        
        if existing_df is None and new_df is None:
            try:
                processed_data_path = os.path.join(folder_path, f"{file_season}_past_fixtures.csv")
                final_dataset_path = os.path.join(folder_path, f"{file_season}_dataset.csv")

                if os.path.exists(final_dataset_path):
                    combined_df = pd.read_csv(final_dataset_path)
                elif os.path.exists(processed_data_path):
                    combined_df = pd.read_csv(processed_data_path)
                else:
                    print(f"No existing data found for season {self.current_season}.")
                    return pd.DataFrame(), pd.DataFrame()
                
                print(f"Loaded data for season {self.current_season}.")
                return pd.DataFrame(), combined_df
            
            except FileNotFoundError:
                print(f"Data for season {self.current_season} not found.")
                return pd.DataFrame(), pd.DataFrame()

        # Fetch current teams data (raw)
        teams_df = self.fetch_current_teams_data()
        teams_df.to_csv(os.path.join(folder_path, f"{file_season}_teams.csv"), index=False)
        if teams_df.empty:
            print("Failed to fetch teams data.")
            new = new_df if new_df is not None else pd.DataFrame()
            existing = existing_df if existing_df is not None else pd.DataFrame()
            combined = pd.concat([existing, new], ignore_index=True)
            return new, combined
        
        processed_existing = pd.DataFrame()
        processed_new = pd.DataFrame()

        if existing_df is not None and not existing_df.empty:
            processed_existing = existing_df.copy()

            if "team_strength" not in processed_existing.columns or self.update_team_metadata:
                processed_existing = append_team_metadata(df=processed_existing, teams_df=teams_df)
                processed_existing = self.process_positions(df=processed_existing)
        
        if new_df is not None and not new_df.empty:
            processed_new = new_df.copy()
            processed_new = append_team_metadata(df=processed_new, teams_df=teams_df)
            processed_new = self.process_positions(df=processed_new)

        if not processed_existing.empty and not processed_new.empty:
            new_gameweeks = set(processed_new.gameweek.unique())
            processed_existing = processed_existing[
                ~((processed_existing.season == processed_existing.season.max()) &
                  (processed_existing.gameweek.isin(new_gameweeks)))
            ]
        combined_df = pd.concat([processed_existing, processed_new], ignore_index=True)

        if self.save_data and not combined_df.empty:
            try:
                os.makedirs(folder_path, exist_ok=True)

                max_gw = combined_df.gameweek.max()
                if max_gw >= 38:
                    processed_file_path = os.path.join(folder_path, f"{file_season}_dataset.csv")
                    print(f"Saving final dataset for season {self.current_season} (GW38 completed).")
                else:
                    processed_file_path = os.path.join(folder_path, f"{file_season}_past_fixtures.csv")

                combined_df.to_csv(processed_file_path, index=False)
                print(f"Processed and saved data for season {self.current_season}.")
                
            except Exception as e:
                print(f"Error saving processed data: {e}")

        return processed_new, combined_df
    

class FPLFutureFixturesFetcher:
    """
    Class for fetching future fixtures data from FPL API
    Follows same pattern as FPLCurrentSeasonFetcher
    """
    def __init__(
            self,
            path: str = "",
            save_data: bool = True,
            current_season: str = "", 
            player_status: list[str] = ["a", "d", "i", "u"],
            horizon: int = 5,
    ):
        """      
        Parameters:
            path: Path for data storage
            save_data: Whether to save processed data to files
            current_season: Current FPL season in 'YY-YY' format
            player_status: Status values to filter players by
            horizon: Number of future gameweeks to fetch
        """
        self.path = path
        self.save_data = save_data
        self.current_season = current_season
        self.player_status = set(player_status)
        self.horizon = horizon
        
        if path:
            os.makedirs(path, exist_ok=True)

    def fetch_fpl_players(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Fetch player and team data from FPL API
        """
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        try:
            time.sleep(random.uniform(0.1, 0.5))
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            players = data["elements"]
            players = [player for player in players if player["status"] in self.player_status]
            teams = data["teams"]
            return players, teams
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return [], []

    def fetch_fixtures(self) -> list[dict[str, Any]]:
        """
        Fetch all fixtures data from FPL API
        """
        url = "https://fantasy.premierleague.com/api/fixtures/"
        try:
            time.sleep(random.uniform(0.1, 0.5))
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch fixtures: {e}")
            return []

    def get_future_gameweeks(self, fixtures: list[dict[str, Any]]) -> list[int]:
        """
        Get the next 'horizon' future gameweeks
        """
        # Filter only future fixtures (those without "finished" status)
        future_fixtures = [fixture for fixture in fixtures if not fixture["finished"]]

        # Check and log cases where "event" is None
        none_event_fixtures = [f for f in future_fixtures if f["event"] is None]
        if none_event_fixtures:
            print(f"Warning: {len(none_event_fixtures)} fixtures have 'event' as None.")

        # Ensure we fetch exactly "horizon" unique gameweeks
        collected_gameweeks = set()
        future_gameweeks = []
        
        for fixture in sorted(future_fixtures, key=lambda x: (x["event"] is None, x["event"])):
            if fixture["event"] is not None and fixture["event"] not in collected_gameweeks:
                collected_gameweeks.add(fixture["event"])
                future_gameweeks.append(fixture["event"])
            if len(collected_gameweeks) == self.horizon:
                break

        print(f"Extracted future gameweeks: {future_gameweeks}")
        print(f"Total unique gameweeks retrieved: {len(future_gameweeks)}")
        
        return future_gameweeks

    def process_player(
            self,
            player: dict[str, Any],
            teams: list[dict[str, Any]], 
            fixtures: list[dict[str, Any]],
            future_gameweeks: list[int],
    ) -> list[dict[str, Any]]:
        """
        Process an individual player's future fixtures data
        """
        player_data = []
        player_id = player["id"]
        player_code = player["code"]
        player_name = player["web_name"]
        team_id = player["team"]
        team_name, team_code = get_team_info(team_id, teams)
        player_status = player.get("status", "")
        
        # Get position name
        position_map = {1: "Goalkeeper", 2: "Defender", 3: "Midfielder", 4: "Forward"}
        player_position = position_map.get(player["element_type"])
        
        if player_position is None:
            return player_data

        try:
            # For each future gameweek, create a fixture entry for this player
            for gameweek in future_gameweeks:
                team_fixture = None
                for fixture in fixtures:
                    if (fixture["event"] == gameweek and 
                        (fixture["team_h"] == team_id or fixture["team_a"] == team_id)):
                        team_fixture = fixture
                        break
                
                if team_fixture:
                    is_home = team_fixture["team_h"] == team_id
                    opponent_team_id = team_fixture["team_a"] if is_home else team_fixture["team_h"]
                    opponent_team_name, opponent_team_code = get_team_info(opponent_team_id, teams)
                    
                    # Get both difficulties
                    opponent_difficulty, team_difficulty, was_home = get_fixture_difficulties(
                        gameweek, team_id, fixtures
                    )
                    
                    # Create gameweek data
                    gameweek_data = {
                        "player_id": player_id,
                        "code": player_code,
                        "player_name": player_name,
                        "team_id": team_id,
                        "team": team_name,
                        "team_code": team_code,
                        "opponent_team_id": opponent_team_id,
                        "opponent_team": opponent_team_name,
                        "opponent_team_code": opponent_team_code,
                        "opponent_difficulty": opponent_difficulty,
                        "team_difficulty": team_difficulty,
                        "was_home": was_home,
                        "season": self.current_season,
                        "status": player_status,
                        "position": player_position,
                        "gameweek": gameweek,
                    }
                    
                    player_data.append(gameweek_data)
                    
        except Exception as e:
            print(f"Error processing player {player_name} (ID: {player_id}): {e}")
        
        return player_data

    def fetch_future_fixtures_data(self) -> pd.DataFrame:
        """
        Fetch future fixtures data for all players
        """
        start_time = time.time()

        players, teams = self.fetch_fpl_players()
        fixtures = self.fetch_fixtures()
        
        if not players or not teams or not fixtures:
            print("Failed to fetch essential data. Aborting.")
            return pd.DataFrame()

        future_gameweeks = self.get_future_gameweeks(fixtures)
        
        if not future_gameweeks:
            print("No future gameweeks found.")
            return pd.DataFrame()

        all_data = []

        # Limit concurrent requests to avoid overwhelming the API
        max_workers = 5
        
        print(f"Processing {len(players)} players for future fixtures...")
        with tqdm(desc="Processing players", total=len(players)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_player = {
                    executor.submit(self.process_player, player, teams, fixtures, future_gameweeks): player
                    for player in players
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_player):
                    player = future_to_player[future]
                    try:
                        player_data = future.result()
                        all_data.extend(player_data)
                    except Exception as e:
                        print(f"Error processing player {player['web_name']}: {e}")
                    finally:
                        pbar.update(1)

        if not all_data:
            print("No future fixtures data found.")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)

        int8_columns = [
            "gameweek", "opponent_difficulty", "team_difficulty", "penalties_order",
            "corners_and_indirect_freekicks_order", "direct_freekicks_order",
        ]

        int16_columns = [
            "now_cost",
        ]

        int32_columns = [
            "player_id", "opponent_team_id", "team_id",
        ]

        int64_columns = [
            "code", "opponent_team_code", "team_code",
        ]

        bool_columns = ["was_home"]

        for col in int8_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int8")

        for col in int16_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int16")

        for col in int32_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int32")

        for col in int64_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int64")

        for col in bool_columns:
            if col in df.columns:
                # Boolean conversion that handles multiple input takes
                df[col] = df[col].map({
                    True: True, "True": True, "true": True, 1: True, "1": True,
                    False: False, "False": False, "false": False, 0: False, "0": False,
                    None: False, "None": False, "": False, np.nan: False
                }).fillna(False).astype("bool")
        
        # Add player metadata
        try:
            player_details_df = get_fpl_players_details(self.path, save_data=False)
            
            if not player_details_df.empty:
                merge_columns = ["code", "now_cost"]
                order_columns = ["corners_and_indirect_freekicks_order", "direct_freekicks_order", "penalties_order"]
                
                # Add order columns if they exist
                for col in order_columns:
                    if col in player_details_df.columns:
                        merge_columns.append(col)
                
                # Merge with player metadata
                df = pd.merge(df, player_details_df[merge_columns], on=["code"], how="left")
                
                # Fill NAs for order columns
                for col in order_columns:
                    if col in df.columns:
                        df[col] = df[col].fillna(5).astype("Int32")
                        
        except Exception as e:
            print(f"Warning: Could not merge player details: {e}")

        if self.save_data:
            folder_season = convert_season_format(self.current_season, target="long_folder")
            file_season = convert_season_format(self.current_season, target="file")
            try:
                folder_path = os.path.join(self.path, folder_season)
                os.makedirs(folder_path, exist_ok=True)
                
                file_path = os.path.join(folder_path, f"{file_season}_raw_future_fixtures.csv")
                df.to_csv(file_path, index=False)
                
                print(f"Saved raw future fixtures data with {len(df)} total rows.")
                
            except Exception as e:
                print(f"Error saving future fixtures data: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"Future fixtures fetch completed in {elapsed_time:.2f} seconds.")
        
        return df


class FPLFutureFixturesProcessor:
    """
    Class for processing future fixtures data with team metadata
    """
    def __init__(
            self,
            path: str = "",
            save_data: bool = True,
            current_season: str = "",
    ):
        """
        Parameters:
            path: Path for data storage
            save_data: Whether to save processed data to files
            current_season: Current FPL season in 'YY-YY' format
        """
        self.path = path
        self.save_data = save_data
        self.current_season = current_season

    def fetch_current_teams_data(self) -> pd.DataFrame:
        """
        Fetch current teams data from FPL API and return raw DataFrame
        """
        try:
            url = "https://fantasy.premierleague.com/api/bootstrap-static/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            teams_df = pd.DataFrame(data["teams"])
            return teams_df
        except Exception as e:
            print(f"Error fetching teams data: {e}")
            return pd.DataFrame()

    def process_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes position fields to long format
        """
        position_dict = {
            "DEF": "Defender",
            "FWD": "Forward",
            "GK": "Goalkeeper",
            "MID": "Midfielder",
            "MNG": "Manager",
        }

        # Check if short-format position names exist
        short_formats = any(pos in df["position"].unique() for pos in position_dict.keys())

        if short_formats:
            df["position"] = df["position"].map(lambda x: position_dict.get(x, x))

        return df

    def process_future_fixtures_data(self, raw_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Process future fixtures data with team metadata
        """
        file_season = self.convert_season_format(self.current_season, target="file")
        long_folder_season = self.convert_season_format(self.current_season, target="long_folder")

        if raw_df is None:
            try:
                folder_path = os.path.join(self.path, long_folder_season)
                raw_data_path = os.path.join(folder_path, f"{file_season}_raw_future_fixtures.csv")
                raw_df = pd.read_csv(raw_data_path)
                print(f"Loaded raw future fixtures data for season {self.current_season}.")
            except FileNotFoundError:
                print(f"Raw future fixtures data for season {self.current_season} not found.")
                return pd.DataFrame()

        if raw_df.empty:
            print(f"No future fixtures data found for season {self.current_season}.")
            return pd.DataFrame()

        # Fetch current teams data (raw)
        teams_df = self.fetch_current_teams_data()
        if teams_df.empty:
            print("Failed to fetch teams data.")
            return raw_df

        season_df = raw_df.copy()

        season_df = append_team_metadata(df=season_df, teams_df=teams_df)
        
        if "position" in season_df.columns:
            season_df = self.process_positions(season_df)

        if self.save_data:
            try:
                folder_path = os.path.join(self.path, long_folder_season)
                os.makedirs(folder_path, exist_ok=True)
                
                processed_file_path = os.path.join(folder_path, f"{file_season}_future_fixtures.csv")
                season_df.to_csv(processed_file_path, index=False)
                
                print(f"Processed and saved future fixtures data for season {self.current_season}.")
                
            except Exception as e:
                print(f"Error saving processed future fixtures data: {e}")

        return season_df
