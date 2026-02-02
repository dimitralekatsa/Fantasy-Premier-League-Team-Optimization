import pandas as pd
import requests
import numpy as np
import os
from fpl.utils.utils import (
    convert_season_format,
    append_team_metadata,
)

class FPLPastSeasonProcessor:
    def __init__(self, path: str = "", save_data: bool = True):
        self.path = path
        self.save_data = save_data
    
    def load_season_data(self, season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads or downloads season data depending on available files
        
        Parameters:
            season: Season in "YY-YY" format
            processed_data: Whether the data are processed or they need to be fetched

        Returns:
            - season_data: Season gameweek data combined
            - fixtures_df: Season fixtures metadata
            - players_raw_df: Season players metadata
            - teams_df: Season teams metadata
        """
        folder_season = convert_season_format(season, target="folder")             # e.g. "2022-23"
        file_season = convert_season_format(season, target="file")                 # e.g. "2022_2023"
        long_folder_season = convert_season_format(season, target="long_folder")   # e.g. "2022-2023"
        
        season_data = pd.DataFrame()

        url = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{folder_season}"

        for i in range(1, 39):
            gw_url = f"{url}/gws/gw{i}.csv"
            try:
                df = pd.read_csv(gw_url)
                if not df.empty:
                    season_data = pd.concat([season_data, df], ignore_index=True, copy=False) if not season_data.empty else df
            except Exception as e:
                print(f"Could not load gameweek {i} for season {season}: {e}")
                continue

        season_data["season"] = season
        season_data = season_data.rename(columns={"round": "gameweek"})

        metadata_files = {
            "fixtures": "fixtures.csv",
            "players_raw": "players_raw.csv",
            "teams": "teams.csv",
        }

        metadata = {}
        for key, filename in metadata_files.items():
            try:
                metadata_df = pd.read_csv(f"{url}/{filename}")
                metadata[key] = metadata_df

                if self.save_data:
                    local_folder = os.path.join(self.path, long_folder_season)
                    os.makedirs(local_folder, exist_ok=True)
                    metadata_df.to_csv(os.path.join(local_folder, f"{file_season}_{filename}"), index=False)

            except Exception as e:
                print(f"Could not load {key} metadata for season {season}: {e}")
                metadata[key] = None

        return season_data, metadata.get("fixtures"), metadata.get("players_raw"), metadata.get("teams")
    
    def map_codes(self, df: pd.DataFrame, players_df: pd.DataFrame, id_col: str = "element") -> pd.DataFrame:
        """
        Maps players identifiers to their codes using the season-specific players DataFrame
        
        Parameters:
            df: DataFrame to map
            players_df: DataFrame containing player codes and season-specific IDs
            id_col: Column in df that holds player IDs. Defaults to "element"
        
        Returns:
            DataFrame with mapped player codes
        """
        id_to_code = players_df.set_index("id")["code"].to_dict()
        df["code"] = df[id_col].map(id_to_code).astype("Int32")

        # Check for any unmapped players
        unmapped_count = df["code"].isna().sum()
        if unmapped_count > 0:
            print(f"Warning: {unmapped_count} players could not be mapped to codes")

        return df
    
    def append_player_metadata(self, df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds static metadata about players to the gameweek data
        
        Parameters:
            df: DataFrame with gameweek data
            players_df: DataFrame containing player metadata

        Returns:
            DataFrame with appended player metadata
        """
        players_by_code = players_df.set_index("code")

        order_columns = ["penalties_order", "corners_and_indirect_freekicks_order", "direct_freekicks_order"]

        for _, col in enumerate(order_columns):
            if col in players_df.columns:
                df[col] = df.code.map(players_by_code[col]).fillna(5).astype("Int16")
            else:
                df[col] = pd.NA

        return df
    
    def append_teams_difficulties(self, df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds opponent difficulty, team difficulty and was_home to the gameweek data of the players

        Parameters:
            df: DataFrame with gameweek data
            fixtures_df: DataFrame containing opponent difficulty ratings

        Returns:
            DataFrame with appended opponent difficulty, team difficulty, and was_home
        """
        opponent_difficulty_map = {}
        team_difficulty_map = {}
        
        # Only create home mapping if was_home column doesn't exist or has missing values
        create_home_mapping = "was_home" not in df.columns or df["was_home"].isna().any()
        if create_home_mapping:
            home_map = {}

        for _, row in fixtures_df.iterrows():
            event = row["event"]
            team_h = row["team_h"]
            team_a = row["team_a"]
            
            opponent_difficulty_map[(event, team_h)] = row["team_a_difficulty"]  # Faces away team
            team_difficulty_map[(event, team_h)] = row["team_h_difficulty"]      # Own difficulty

            opponent_difficulty_map[(event, team_a)] = row["team_h_difficulty"]  # Faces home team
            team_difficulty_map[(event, team_a)] = row["team_a_difficulty"]      # Own difficulty
            
            if create_home_mapping:
                home_map[(event, team_h)] = True
                home_map[(event, team_a)] = False

        df["opponent_difficulty"] = df.apply(
            lambda x: opponent_difficulty_map.get((x["gameweek"], x["team_id"])), axis=1
        ).astype("Int64")
        
        df["team_difficulty"] = df.apply(
            lambda x: team_difficulty_map.get((x["gameweek"], x["team_id"])), axis=1
        ).astype("Int64")

        if create_home_mapping:
            df["was_home"] = df.apply(
                lambda x: home_map.get((x["gameweek"], x["team_id"])), axis=1
            ).astype("bool")

        return df
    
    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the DataFrame by removing players and teams identifiers that change every season (e.g., 'id', 'team_id')
        """
        cols_to_drop = [
            "name", "element", "team", "team_id", "opponent_team",
        ]

        cols_to_drop = [col for col in cols_to_drop if col in df.columns]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def process_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes position fields to long format (Defender, Midfielder, Goalkeeper, Forward)
        """
        position_dict = {
            "DEF": "Defender",
            "FWD": "Forward",
            "GK": "Goalkeeper",
            "GKP": "Goalkeeper",
            "MID": "Midfielder",
            "MNG": "Manager",
        }

        # Check if long-format position names exist
        short_formats = any(pos in df["position"].unique() for pos in position_dict.keys())

        if short_formats:
            # Map long-format position names to short-format
            df["position"] = df["position"].map(lambda x: position_dict.get(x, x))

        return df
    
    def process_season_data(self, season: str, processed_data: bool = True) -> pd.DataFrame:
        """
        Process data for a specific season
        """
        file_season = convert_season_format(season, target="file")                 # e.g. "2022_2023"
        long_folder_season = convert_season_format(season, target="long_folder")   # e.g. "2022-2023"

        if processed_data:
            try:
                season_df = pd.read_csv(os.path.join(self.path, long_folder_season, f"{file_season}_dataset.csv"))
                print(f"Loaded pre-processed data for season {season}.")
                return season_df
            except FileNotFoundError:
                print(f"Pre-processed data for season {season} not found. Fetching new data...")
                processed_data = False

        if not processed_data:
            raw_df, fixtures_data, players_raw_data, teams_data = self.load_season_data(season)
            if raw_df.empty:
                print(f"No data found for season {season}.")
                return pd.DataFrame()

            try:
                players_raw_df = pd.read_csv(os.path.join(self.path, long_folder_season, f"{file_season}_players_raw.csv"))
            except FileNotFoundError:
                print(f"Players metadata for season {season} not found. Fetching new data...")
                players_raw_df = None
            
            if players_raw_df is None:
                players_raw_df = players_raw_data.copy()

            try:
                teams_df = pd.read_csv(os.path.join(self.path, long_folder_season, f"{file_season}_teams.csv"))
            except FileNotFoundError:
                print(f"Teams data for season {season} not found. Fetching new data...")
                teams_df = None

            if teams_df is None:
                teams_df = teams_data.copy()

            try:
                fixtures_df = pd.read_csv(os.path.join(self.path, long_folder_season, f"{file_season}_fixtures.csv"))
            except FileNotFoundError:
                print(f"Fixtures data for season {season} not found. Fetching new data...")
                fixtures_df = None

            if fixtures_df is None:
                fixtures_df = fixtures_data.copy()
            
            season_df = raw_df.copy()

            if "code" not in season_df.columns and players_raw_df is not None:
                season_df = self.map_codes(season_df, players_raw_df)               # Map players codes
            elif players_raw_df is None:
                print(f"Cannot map codes for season {season}: players metadata not found.")
                return pd.DataFrame()

            if players_raw_df is not None:
                season_df = self.append_player_metadata(season_df, players_raw_df)  # Append player metadata

            if teams_df is not None:
                season_df = append_team_metadata(season_df, teams_df)               # Append team metadata

            if fixtures_df is not None:
                season_df = self.append_teams_difficulties(season_df, fixtures_df)  # Append teams difficulties

            season_df = self.process_positions(season_df)

            season_df = self.clean_df(season_df)

            int8_columns = [
                "assists", "bonus", "clean_sheets", "goals_conceded", "goals_scored", "own_goals",
                "penalties_missed", "penalties_saved", "red_cards", "gameweek", "yellow_cards",
                "team_a_score", "team_h_score", "penalties_order", "corners_and_indirect_freekicks_order",
                "direct_freekicks_order", "opponent_difficulty", "team_difficulty", "saves",
            ]

            int16_columns = [
                "bps", "fixture", "minutes", "threat", "total_points", "value", 
            ]

            int32_columns = [
                "team_strength", "team_strength_overall", "team_strength_attack", "team_strength_defence",
            ]

            int64_columns = [
                "code", "team_code", "opponent_team_code", 
            ]

            float_columns = [
                "creativity", "expected_assists", "expected_goal_involvements", "expected_goals",
                "expected_goals_conceded", "ict_index", "influence", 
            ]

            bool_columns = ["was_home"]

            for col in int8_columns:
                if col in season_df.columns:
                    season_df[col] = pd.to_numeric(season_df[col], errors="coerce").fillna(0).astype("Int8")

            for col in int16_columns:
                if col in season_df.columns:
                    season_df[col] = pd.to_numeric(season_df[col], errors="coerce").fillna(0).astype("Int16")

            for col in int32_columns:
                if col in season_df.columns:
                    season_df[col] = pd.to_numeric(season_df[col], errors="coerce").fillna(0).astype("Int32")

            for col in int64_columns:
                if col in season_df.columns:
                    season_df[col] = pd.to_numeric(season_df[col], errors="coerce").fillna(0).astype("Int64")

            for col in float_columns:
                if col in season_df.columns:
                    season_df[col] = pd.to_numeric(season_df[col], errors="coerce").fillna(0.0).astype("float32")

            for col in bool_columns:
                if col in season_df.columns:
                    # Boolean conversion that handles multiple input takes
                    season_df[col] = season_df[col].map({
                        True: True, "True": True, "true": True, 1: True, "1": True,
                        False: False, "False": False, "false": False, 0: False, "0": False,
                        None: False, "None": False, "": False, np.nan: False
                    }).fillna(False).astype("bool")

            if self.save_data:
                season_df.to_csv(os.path.join(self.path, long_folder_season, f"{file_season}_dataset.csv"), index=False)
                print(f"Processed and saved data for season {season}.")

            return season_df
        
    def process_multiple_seasons(self, seasons: list | None = None, processed_data: bool = True) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Process multiple seasons at once
        
        Parameters:
            seasons: List of seasons to process (e.g., ["22-23", "23-24"])
            
        Returns:
            - Dictionary of processed DataFrames keyed by season
            - Combined dataset of all seasons
        """
        if seasons is None:
            seasons = ["22-23", "23-24", "24-25"]  # Default seasons to process
            
        processed_seasons = {}
        for season in seasons:
            processed_seasons[season] = self.process_season_data(season, processed_data=processed_data)
            
        # Combine all seasons into one DataFrame
        if all(not df.empty for df in processed_seasons.values()):
            combined_df = pd.concat(list(processed_seasons.values()), ignore_index=True)
            return processed_seasons, combined_df
        else:
            # Return empty DataFrame if any season failed to process
            empty_seasons = [s for s, df in processed_seasons.items() if df.empty]
            print(f"Warning: Could not process seasons {empty_seasons}")
            non_empty_dfs = [df for df in processed_seasons.values() if not df.empty]
            if non_empty_dfs:
                combined_df = pd.concat(non_empty_dfs, ignore_index=True)
                return processed_seasons, combined_df
            else:
                return processed_seasons, pd.DataFrame()
