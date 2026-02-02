import pandas as pd
import numpy as np
import requests
import os
from typing import Any

def get_fpl_players_details(path: str = "", save_data: bool = True) -> pd.DataFrame:
    """
    Fetch current FPL player details from the API
    
    Parameters:
        path: path to save the data
        save_data: whether to save the data to CSV file
        
    Returns:
        DataFrame containing player details
    """
    try:
        response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        summary = response.json()
        
        # Get players data
        players = pd.DataFrame(summary["elements"])
        
        # Merge player position names from element_types
        positions = pd.DataFrame(summary["element_types"])
        players = players.merge(positions[["id", "singular_name"]], 
                               left_on="element_type", right_on="id", how="left")
        
        # Fix the column renaming
        players = players.rename(columns={
            "singular_name": "position",
            "id_x": "player_id",
            "goals_scored": "Total_goals_scored",
            "assists": "Total_assists",
            "minutes": "Total_minutes",
            "total_points": "Season_points",
            "clean_sheets": "Total_clean_sheets",
            "yellow_cards": "Total_yellow_cards",
            "red_cards": "Total_red_cards",
            "own_goals": "Total_own_goals",
            "penalties_saved": "Total_penalties_saved",
            "penalties_missed": "Total_penalties_missed",
            "bonus": "Total_bonus_points",
            "influence": "Total_influence",
            "creativity": "Total_creativity",
            "threat": "Total_threat",
            "ict_index": "Total_ict_index",
            "expected_goals": "Season_expected_goals",
            "expected_assists": "Season_expected_assists",
            "expected_goals_conceded": "Season_expected_goals_conceded",
        })
        
        players = players.drop("id_y", axis=1)

        if save_data and path:
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, "player_details.csv")
            players.to_csv(file_path, index=False)
            print(f"Player details saved to {file_path}")        
        return players
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player details from API: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    except Exception as e:
        print(f"Error processing player details: {e}")
        return pd.DataFrame()

def convert_season_format(season: str, target: str = "folder",) -> str:
    """
    Convert season string from "YY-YY" to other formats

    Parameters:
        season: Season string in "YY-YY" format
        target: Target format - "folder": "20YY-YY", "long_folder": "20YY-20YY", "file": "20YY_20YY", "short_file": "YY_YY"
    """
    start, end = season.split("-")
    start_full = f"20{start}"
    end_full = f"20{end}"

    if target == "folder":
        return f"{start_full}-{end}"
    elif target == "long_folder":
        return f"{start_full}-{end_full}"
    elif target == "file":
        return f"{start_full}_{end_full}"
    elif target == "short_file":
        return f"{start}_{end}"
    
    return season

def get_team_info(team_id: int, teams: list[dict[str, Any]]) -> tuple[str | None, int | None]:
    """
    Get team name and code from team ID
    """
    for team in teams:
        if team["id"] == team_id:
            return team["name"], team["code"]
    return None, None

def get_fixture_difficulties(
        gameweek: int,
        team_id: int,
        fixtures: list[dict[str, Any]],
) -> tuple[int | None, int | None, bool | None]:
    """
    Get both team's difficulties for a fixture

    Returns:
        (team_difficulty, opponent_difficulty, was_home)
    """
    for fixture in fixtures:
        if fixture["event"] == gameweek:
            if fixture["team_h"] == team_id:
                return fixture["team_h_difficulty"], fixture["team_a_difficulty"], True
            elif fixture["team_a"] == team_id:
                return fixture["team_a_difficulty"], fixture["team_h_difficulty"], False
    return None, None, None

def append_team_metadata(df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches player data with team statistics and team identifiers
    
    Parameters:
        df: Input DataFrame with gameweek data
        teams_df: DataFrame containing team metadata
    """
    # Create mapping dictionaries from raw teams data
    name_to_code = teams_df.set_index("name")["code"].to_dict()
    code_to_id = teams_df.set_index("code")["id"].to_dict()
    id_to_code = teams_df.set_index("id")["code"].to_dict()
    teams_by_code = teams_df.set_index("code")

    # Map the team codes to the players DataFrame
    df["team_code"] = df["team"].map(name_to_code).astype("Int32")
    df["team_id"] = df["team_code"].map(code_to_id).astype("Int32")
    df["opponent_team_code"] = df["opponent_team_id"].map(id_to_code).astype("Int32")

    # Add team statistics
    df["team_strength"] = df["team_code"].map(teams_by_code["strength"])

    df["team_strength_overall"] = np.where(
        df["was_home"],
        df["team_code"].map(teams_by_code["strength_overall_home"]),
        df["team_code"].map(teams_by_code["strength_overall_away"])
    )

    df["team_strength_attack"] = np.where(
        df["was_home"],
        df["team_code"].map(teams_by_code["strength_attack_home"]),
        df["team_code"].map(teams_by_code["strength_attack_away"]),
    )

    df["team_strength_defence"] = np.where(
        df["was_home"],
        df["team_code"].map(teams_by_code["strength_defence_home"]),
        df["team_code"].map(teams_by_code["strength_defence_away"]),
    )

    return df

def handle_double_gameweeks(df: pd.DataFrame, points_column_name: str = "predicted_points") -> pd.DataFrame:
    """
    Handle double gameweeks by summing predicted points and aggregating other columns appropriately
    
    Parameters:
        df: input dataframe with potential double gameweeks
        points_column_name: string to search for in column names
    
    Returns:
        result_df: Dataframe with double gameweeks properly aggregated
    """

    # Check if we have potential double gameweeks (multiple rows per player per gameweek)
    group_cols = ["player_id", "gameweek"] if "player_id" in df.columns else ["code", "gameweek"]
    duplicates = df.groupby(group_cols).size()
    has_doubles = (duplicates > 1).any()

    if not has_doubles:
        print("No double gameweeks detected.")
        return df
    
    print(f"Double gameweeks detected for {(duplicates > 1).sum()} player-gameweek combinations.")

    # Find points columns based on prefix
    if points_column_name is None:
        points_columns = [col for col in df.columns if "predicted_points" in col.lower()]
        print(f"Detected points columns: {points_columns}")
    else:
        points_columns = [col for col in df.columns if points_column_name in col]
        if not points_columns:
            print(f"Warning: No columns found containing '{points_column_name}'. Using auto-detection.")
            points_columns = [col for col in df.columns if "predicted" in col.lower()]
            print(f"Auto-detected points columns: {points_columns}")      
            
    agg_dict = {}

    # Sum all found points columns
    if "total_points" in df.columns:
        points_columns.append("total_points")
    for col in points_columns:
        agg_dict[col] = "sum"

    # Standard columns that should be taken as "first"
    first_columns = [
        "player_name", "position", "team", "team_id", "team_code", 
        "now_cost", "status", "opponent_team", "opponent_team_id", "opponent_team_code", "was_home",
    ]
    if "player_id" in df.columns:
        first_columns.append("code")
    for col in first_columns:
        if col in df.columns:
            agg_dict[col] = "first"

    # For double gameweeks, we take the maximum difficulty
    max_columns = ["opponent_difficulty", "team_difficulty"]
    for col in max_columns:
        if col in df.columns:
            agg_dict[col] = "max"

    result_df = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    id_cols = ["player_id", "team_id", "opponent_team_id"]
    for col in id_cols:
        result_df[col] = result_df[col].astype("Int16")

    id_columns = ["player_id", "team_id", "opponent_team_id", "team_code", "opponent_team_code"]
    for col in id_columns:
        if col in result_df.columns:
            result_df[col] = result_df[col].astype(int)
    
    return result_df

def transform_to_wide_format(df: pd.DataFrame, points_column_name: str = "predicted_points", prefix: str = "Points_GW") -> pd.DataFrame:
    """
    Transform the long-format FPL dataframe to a wide format with gameweeks as columns for transfer enginess
    
    Parameters:
        df: original dataframe in long format
        points_column_name: name of the column containing the predicted points
        prefix: prefix for the new gameweek columns
    
    Returns:
        wide_df: wide format dataframe with one row per player and columns for each gameweek
    """
    
    # Identify the player attributes that should be preserved (non-gameweek specific)
    player_attrs = [
        "player_id", "code", "player_name", "team", "team_id", "team_code", "opponent_team",
        "opponent_team_id", "opponent_team_code", "position", "status", "now_cost", "in_team",
        "team_difficulty", "opponent_difficulty", "total_points",
    ]
    
    # Make sure all these columns exist, otherwise remove them from the list
    player_attrs = [col for col in player_attrs if col in df.columns]
    
    # Get unique players and their attributes
    player_info = df.drop_duplicates(subset=["player_id"]).loc[:, player_attrs]
    
    # Get all unique gameweeks in the prediction horizon
    all_gameweeks = sorted(df["gameweek"].unique())
    
    # Create the wide dataframe for predicted points
    points_wide = df.pivot(index="player_id", columns="gameweek", values=points_column_name)
    
    # Rename the columns to include the prefix
    points_wide.columns = [f"{prefix}{gw}" for gw in points_wide.columns]
    
    # Merge player info with the points
    wide_df = player_info.set_index("player_id").join(points_wide)
    
    # Reset index to make player_id a regular column
    wide_df = wide_df.reset_index()
    
    # Ensure all players have columns for all gameweeks (even if NaN)
    for gw in all_gameweeks:
        col_name = f"{prefix}{gw}"
        if col_name not in wide_df.columns:
            wide_df[col_name] = float("nan")
    
    # Sort columns: player attributes first, then gameweek columns in order
    gw_cols = sorted([col for col in wide_df.columns if col.startswith(prefix)], 
                    key=lambda x: int(x.replace(prefix, "")))
    other_cols = [col for col in wide_df.columns if not col.startswith(prefix)]
    
    wide_df = wide_df[other_cols + gw_cols]

    id_cols = ["player_id", "team_id", "opponent_team_id"]
    for col in id_cols:
        wide_df[col] = wide_df[col].astype("Int16")
    
    return wide_df

def get_current_gameweek():
    """
    Get current FPL gameweek number
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    data = response.json()
    current_gw = next((event["id"] for event in data["events"] if event["is_current"]), 1)
    return current_gw

def backfill_missing_players(df: pd.DataFrame, missing_players: list, prefix: str = "Points_GW") -> tuple[pd.DataFrame, bool]:
    """
    Backfill missing players in the predictions dataframe with zero predicted points

    Parameters:
        df: DataFrame containing player predictions
        missing_players: list of player IDs that are missing and need to be backfilled
        prefix: prefix used for gameweek points columns

    Returns:
        (updated_df, found_all)
    """
    if not missing_players:
        return df, True
    
    preds_col = [col for col in df.columns if col.startswith(prefix)]

    training_path = os.path.join("data", "training_dataset.csv")
    if not os.path.exists(training_path):
        print(f"Training data file not found at {training_path}. Cannot backfill missing players.")
        return df, False
    
    train_df = pd.read_csv(training_path)
    
    required_columns = ["player_id", "player_name", "code","team", "team_id", "team_code", "position", "status", "now_cost"]

    missing_cols = [c for c in required_columns if c not in train_df.columns]
    if missing_cols:
        print(f"Training data is missing required columns: {missing_cols}. Cannot backfill missing players.")
        return df, False
    
    seasons = df["season"].unique()
    if len(seasons) != 1:
        print("DataFrame contains multiple seasons. Cannot backfill missing players.")
        return df, False
    season_value = seasons[0]

    for pid in missing_players:
        player_data = train_df[train_df["player_id"] == pid]
        if player_data.empty:
            return df, False
               
        player_season_data = player_data[player_data.season == season_value]
        
        if player_season_data.empty:
            return df, False
        
        player_season_data = player_season_data.sort_values(by="gameweek")
        player_info = player_season_data.iloc[-1][required_columns].to_dict()
        
        # Create a new row with zero predicted points
        new_row = {col: 0 for col in preds_col}
        new_row.update(player_info)
        
        # Append the new row to the dataframe
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Backfilled missing player ID {pid} - {player_info.get('player_name', 'Unknown')}")

    return df, True
