import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional

class FPLPastFeatureEngineering:
    """
    FPL data preprocessing class that handles feature engineering for multiple seasons datasets
    In all functions, clauses that check if they features have already been calculated exist,
    to avoid recalculations of already processed data
    """
    def __init__(self, path: str = "", save_data: bool = True):
        """
        Initialize FPL preprocesing pipeline.

        Parameters:
            path (str): Base path for data storage
            save_data (bool): Whether to save processed data to files
        """
        self.path = path
        self.save_data = save_data

    def calculate_averages_difficulty(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Calculate difficulty-adjusted rolling averages with weighted combination of current-past season data
        """
        # Sort by player, season, and gameweek to ensure correct rolling calculation
        df = df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)

        numeric_cols = [
            "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", 
            "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", 
            "red_cards", "saves", "bonus", "bps", "total_points", "threat", "ict_index", 
            "influence", "creativity", "expected_goals", "expected_assists", 
            "expected_goal_involvements", "expected_goals_conceded",
        ]

        # Initialize average columns if they do not exist
        for col in numeric_cols:
            if f"{col}_average_difficulty_{window}" not in df.columns:
                df[f"{col}_average_difficulty_{window}"] = np.nan

        # Detect which data need processing
        if "expected_goals" in df.columns:
            needs_processing = df[f"{numeric_cols[-1]}_average_difficulty_{window}"].isna()
        else:
            needs_processing = df[f"{numeric_cols[1]}_average_difficulty_{window}"].isna()

        if not needs_processing.any():
            return df

        seasons = sorted(df["season"].unique())

        # Iterate over each player to calculate the means by difficulty independently for each season
        for player_id in df.code.unique():
            player_full_df = df[df.code == player_id].copy()

            # Get Nan rows for each specific player
            player_needs_processing = player_full_df[f"{numeric_cols[1]}_average_difficulty_{window}"].isna()
            player_df = player_full_df[player_needs_processing].copy()

            # Skip player with no rows needing processing
            if player_df.empty:
                continue

            for idx in player_df.index:
                current_season = df.loc[idx, "season"]
                current_gw = df.loc[idx, "gameweek"]
                current_diff = df.loc[idx, "opponent_difficulty"]

                current_season_df = player_full_df[player_full_df.season == current_season]

                # Case 1: Player's first season
                if current_season == player_full_df.season.iloc[0]:
                    # Case 1A: first row
                    if idx == current_season_df.index[0]:
                        for col in numeric_cols:
                            df.loc[idx, f"{col}_average_difficulty_{window}"] = float(df.loc[idx, col])
                        continue

                    # Same difficulty data for previous gws
                    diff_df = current_season_df[
                        (current_season_df.opponent_difficulty == current_diff) &
                        (current_season_df.gameweek < current_gw)
                    ]

                    # Case 1B: same difficulty data exists
                    if not diff_df.empty:
                        for col in numeric_cols:
                            df.loc[idx, f"{col}_average_difficulty_{window}"] = float(diff_df[col].tail(window).mean())
                    # Case 1C: same difficulty data does not exist - use gerneral rolling average
                    else:
                        prev_data = current_season_df[current_season_df.gameweek < current_gw].tail(window)
                        for col in numeric_cols:
                            if not prev_data.empty:
                                rolling_avg = prev_data[col].mean()
                                df.loc[idx, f"{col}_average_difficulty_{window}"] = float(rolling_avg)
                            else:
                                df.loc[idx, f"{col}_average_difficulty_{window}"] = float(df.loc[idx, col])

                # Case 2: Player's subsequent seasons
                else:
                    current_season_idx = seasons.index(current_season)
                    previous_season_df = pd.DataFrame()

                    for i in range(current_season_idx - 1, -1, -1):
                        temp_df = player_full_df[player_full_df.season == seasons[i]]
                        if not temp_df.empty:
                            previous_season_df = temp_df
                            break

                    # Same difficulty data for previous gws
                    current_diff_df = current_season_df[
                        (current_season_df.opponent_difficulty == current_diff) &
                        (current_season_df.gameweek < current_gw)
                    ]

                    for col in numeric_cols:
                        # Case 2A: sufficient same difficulty data in current season
                        if len(current_diff_df) >= window:
                            df.loc[idx, f"{col}_average_difficulty_{window}"] = float(current_diff_df[col].tail(window).mean())

                        # Case 2B: insufficient same difficulty data in current season
                        elif len(current_diff_df) > 0:
                            n_current = len(current_diff_df)
                            current_weight = n_current / window
                            past_weight = 1.0 - current_weight
                            past_data_needed = window - n_current

                            current_avg = current_diff_df[col].mean()

                            if not previous_season_df.empty:
                                past_diff_df = previous_season_df[previous_season_df.opponent_difficulty == current_diff].tail(past_data_needed)

                                # Case 2B1: past same difficulty data exists
                                if not past_diff_df.empty:
                                    past_avg = past_diff_df[col].mean()
                                    weighted_avg = current_weight * current_avg + past_weight * past_avg
                                    df.loc[idx, f"{col}_average_difficulty_{window}"] = float(weighted_avg)
                                # Case 2B2: no past same difficulty data exists - use general past data
                                else:
                                    past_general_df = previous_season_df.tail(past_data_needed)
                                    if not past_general_df.empty:
                                        past_avg = past_general_df[col].mean()
                                        weighted_avg = current_weight * current_avg + past_weight * past_avg
                                        df.loc[idx, f"{col}_average_difficulty_{window}"] = float(weighted_avg)
                                    else:
                                        df.loc[idx, f"{col}_average_difficulty_{window}"] = float(current_avg)
                            
                            # No past data exists
                            else:
                                df.loc[idx, f"{col}_average_difficulty_{window}"] = float(current_avg)

                        # Case 2C: no same difficulty data exists
                        else:
                            if not previous_season_df.empty:
                                past_diff_df = previous_season_df[previous_season_df.opponent_difficulty == current_diff].tail(window)
                                # Case 2C1: past same difficulty data exists
                                if not past_diff_df.empty:
                                    df.loc[idx, f"{col}_average_difficulty_{window}"] = float(past_diff_df[col].mean())
                                # Case 2C2: no past same difficulty data exists - use general data
                                else:
                                    general_current = current_season_df[current_season_df.gameweek < current_gw]

                                    if len(general_current) >= window:
                                        rolling_avg = general_current[col].tail(window).mean()
                                        df.loc[idx, f"{col}_average_difficulty_{window}"] = float(rolling_avg)
                                    elif len(general_current) > 0:
                                        n_current = len(general_current)
                                        current_weight = n_current / window
                                        past_weight = 1.0 - current_weight
                                        past_data_needed = window - n_current

                                        current_avg = general_current[col].mean()
                                        past_general_df = previous_season_df.tail(past_data_needed)

                                        if not past_general_df.empty:
                                            past_avg = past_general_df[col].mean()
                                            weighted_avg = current_weight * current_avg + past_weight * past_avg
                                            df.loc[idx, f"{col}_average_difficulty_{window}"] = float(weighted_avg)
                                        else:
                                            df.loc[idx, f"{col}_average_difficulty_{window}"] = float(current_avg)
                                    else:
                                        past_general_df = previous_season_df.tail(window)
                                        if not past_general_df.empty:
                                            df.loc[idx, f"{col}_average_difficulty_{window}"] = float(past_general_df[col].mean())
                                        else:
                                            df.loc[idx, f"{col}_average_difficulty_{window}"] = float(df.loc[idx, col])

                            # No previous season data - use general current season data
                            else:
                                general_current = current_season_df[current_season_df.gameweek < current_gw]
                                if not general_current.empty:
                                    df.loc[idx, f"{col}_average_difficulty_{window}"] = float(general_current[col].tail(window).mean())
                                else:   # No general current data exists - use current row value
                                    df.loc[idx, f"{col}_average_difficulty_{window}"] = float(df.loc[idx, col])

        return df
    
    def calculate_rolling_averages(self, df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
        """
        Calculate rolling averages over a specified window with weighted combination of current and previous data
        """
        numeric_cols = [
            "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", "own_goals",
            "penalties_saved", "penalties_missed", "yellow_cards", "red_cards", "saves", "bonus", "bps",
            "total_points", "threat", "ict_index", "influence", "creativity", "expected_goals",
            "expected_assists", "expected_goal_involvements", "expected_goals_conceded",
        ]

        df = df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)

        for col in numeric_cols:
            rolling_col = f"{col}_average_rolling_{window}"
            if rolling_col not in df.columns:
                df[rolling_col] = np.nan

        # Detect which data need processing
        if "expected_goals" in df.columns:
            needs_processing = df[f"{numeric_cols[-1]}_average_rolling_{window}"].isna()
        else:
            needs_processing = df[f"{numeric_cols[1]}_average_rolling_{window}"].isna()

        if not needs_processing.any():
            return df
        
        seasons = sorted(df["season"].unique())

        for player_id in df.code.unique():
            player_full_df = df[df.code == player_id].copy()

            player_needs_processing = player_full_df[f"{numeric_cols[1]}_average_rolling_{window}"].isna()
            player_df = player_full_df[player_needs_processing].copy()

            if player_df.empty:
                continue

            for idx in player_df.index:
                current_season = df.loc[idx, "season"]
                current_gw = df.loc[idx, "gameweek"]

                current_season_df = player_full_df[player_full_df.season == current_season]

                # Case 1: Player's first season
                if current_season == player_full_df.season.iloc[0]:
                    # Case 1A: First row of the oldest season
                    if idx == current_season_df.index[0]:
                        for col in numeric_cols:
                            df.loc[idx, f"{col}_average_rolling_{window}"] = float(df.loc[idx, col])
                        continue

                    # Case 1B: Simple rolling average for current season
                    for col in numeric_cols:
                        prev_data = current_season_df[current_season_df.gameweek < current_gw][col]

                        if not prev_data.empty:
                            rolling_avg = prev_data.tail(window).mean()
                            df.loc[idx, f"{col}_average_rolling_{window}"] = float(rolling_avg)
                        else:
                            df.loc[idx, f"{col}_average_rolling_{window}"] = float(df.loc[idx, col])

                # Case 2: Player's subsequent seasons
                else:
                    current_season_idx = seasons.index(current_season)
                    previous_season_df = pd.DataFrame()

                    for i in range(current_season_idx - 1, -1, -1):
                        temp_df = player_full_df[player_full_df.season == seasons[i]]
                        if not temp_df.empty:
                            previous_season_df = temp_df
                            break

                    current_data = current_season_df[current_season_df.gameweek < current_gw]

                    for col in numeric_cols:
                        # Case 2A: Sufficient current season data
                        if len(current_data) >= window:
                            current_avg = current_data[col].tail(window).mean()
                            df.loc[idx, f"{col}_average_rolling_{window}"] = float(current_avg)

                        # Case 2B: Insufficient current season data - weighted average logic
                        else:
                            n_current = len(current_data)
                            weight_current = n_current / window
                            weight_past = 1.0 - weight_current
                            past_data_needed = window - n_current

                            if n_current > 0:
                                current_avg = current_data[col].mean()
                            else:
                                current_avg = 0.0   # No current data

                            if not previous_season_df.empty:
                                past_data = previous_season_df[col].tail(past_data_needed)
                                if not past_data.empty:
                                    past_avg = past_data.mean()
                                    weighted_avg = weight_current * current_avg + weight_past * past_avg
                                    df.loc[idx, f"{col}_average_rolling_{window}"] = float(weighted_avg)
                                else:
                                    df.loc[idx, f"{col}_average_rolling_{window}"] = float(current_avg)
                            else:
                                df.loc[idx, f"{col}_average_rolling_{window}"] = float(current_avg)

        return df

    def calculate_form_and_average_minutes(self, df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
        """
        Calculate rolling averages for form (total_points) and minutes with normalization
        Normalization happens to only newly processed data to reserve consistency in the dataset
        """
        # Sort data to ensure correct rolling calculation
        df = df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)

        new_columns = [
            f"form_average_rolling_{window}",
            f"form_average_rolling_{window}_normalized",
            f"form_average_rolling_{window}_sqrt_normalized", 
            f"minutes_average_rolling_{window}",
            f"minutes_average_rolling_{window}_normalized",
        ]

        for col in new_columns:
            if col not in df.columns:
                df[col] = np.nan

        # Detect which data need processing
        needs_processing = df[new_columns[0]].isna()
        if not needs_processing.any():
            return df
        
        seasons = sorted(df["season"].unique())

        # Case 1: sufficient current season data
        df.loc[needs_processing, f"form_average_rolling_{window}"] = (
            df.groupby(["code", "season"])["total_points"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .astype("float64")
            .loc[needs_processing]
        )
        
        df.loc[needs_processing, f"minutes_average_rolling_{window}"] = (
            df.groupby(["code", "season"])["minutes"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
            .astype("float64")
            .loc[needs_processing]
        )

        # Case 2: first gameweek fallback
        for player_id in df.code.unique():
            player_data = df[df.code == player_id]
            player_seasons = sorted(player_data.season.unique())

            for season in player_seasons:
                season_idx = seasons.index(season)
                player_season_data = player_data[player_data.season == season]
                first_gw = player_season_data.gameweek.min()
                
                first_gw_mask = (
                    (df.code == player_id) & 
                    (df.season == season) & 
                    (df.gameweek == first_gw)
                )
                
                # Case 2A: Use previous season data
                if season != player_seasons[0]:
                    previous_season_data = pd.DataFrame()

                    for i in range(season_idx - 1, -1, -1):
                        temp_season = seasons[i]
                        temp_data = player_data[player_data.season == temp_season]
                        if not temp_data.empty:
                            previous_season_data = temp_data
                            break
                    
                    if not previous_season_data.empty:
                        prev_season_tail = previous_season_data.tail(window)
                        form_avg = prev_season_tail.total_points.mean()
                        minutes_avg = prev_season_tail.minutes.mean()
                        
                        df.loc[first_gw_mask, f"form_average_rolling_{window}"] = float(form_avg)
                        df.loc[first_gw_mask, f"minutes_average_rolling_{window}"] = float(minutes_avg)
                        continue
                
                # Case 2B: player's first appearance - use current values
                df.loc[first_gw_mask, f"form_average_rolling_{window}"] = df.loc[first_gw_mask, "total_points"].astype("float64")
                df.loc[first_gw_mask, f"minutes_average_rolling_{window}"] = df.loc[first_gw_mask, "minutes"].astype("float64")

        newly_processed = (
                df[f"minutes_average_rolling_{window}"].notna() &
                df[f"minutes_average_rolling_{window}_normalized"].isna()
            )

        if newly_processed.any():
            # Minutes normalization (divide by 90)
            minutes_col = f"minutes_average_rolling_{window}"
            normalized_minutes_col = f"minutes_average_rolling_{window}_normalized"
            df.loc[newly_processed, normalized_minutes_col] = (df.loc[newly_processed, minutes_col] / 90).astype("float64")

            # Form normalization
            form_col = f"form_average_rolling_{window}"
            df_sorted = df.sort_values(by=["season", "gameweek"]).reset_index(drop=True)

            for season in seasons:
                season_mask_all = (df_sorted["season"] == season)
                season_data_all = df_sorted[season_mask_all].copy()

                season_mask_process = newly_processed & (df_sorted["season"] == season)

                if not season_mask_process.any():
                    continue

                expanding_max = season_data_all[form_col].expanding().max()
                form_col_safe = np.maximum(season_data_all[form_col], 0)                # Avoid negative values for sqrt normalization
                expanding_sqrt_max = np.sqrt(form_col_safe).expanding().max()

                # Handle first gameweek: use current form as max (normalized to 1)
                first_gw = season_data_all.gameweek.min()
                first_gw_mask = season_data_all.gameweek == first_gw
                expanding_max[first_gw_mask] = season_data_all.loc[first_gw_mask, form_col]
                expanding_sqrt_max[first_gw_mask] = np.sqrt(np.maximum(season_data_all.loc[first_gw_mask, form_col], 0))

                # Handle remaining data
                expanding_max = expanding_max.fillna(0.1).replace(0, 0.1)               # Avoid division by 0
                expanding_sqrt_max = expanding_sqrt_max.fillna(0.1).replace(0, 0.1)     # Avoid division by 0

                process_indices = df_sorted[season_mask_process].index

                df.loc[process_indices, f"form_average_rolling_{window}_normalized"] = (
                    season_data_all.loc[process_indices, form_col] / expanding_max.loc[process_indices]
                ).astype("float64")

                df.loc[process_indices, f"form_average_rolling_{window}_sqrt_normalized"] = (
                    np.sqrt(np.maximum(season_data_all.loc[process_indices, form_col], 0)) / expanding_sqrt_max.loc[process_indices]
                ).astype("float64")
        
        return df

    def add_team_goals_scored_conceded(self, df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
        """
        Adds goals scored/conceded by the player's team and its opponent
        """

        df = df.sort_values(by=["season", "team_code", "gameweek"]).reset_index(drop=True)

        new_columns = [
            f"team_goals_scored_rolling_{window}",
            f"team_goals_conceded_rolling_{window}",
            f"opponent_goals_scored_rolling_{window}",
            f"opponent_goals_conceded_rolling_{window}",
        ]

        for col in new_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        needs_processing = df[new_columns[0]].isna()
        if not needs_processing.any():
            return df

        # Process team's goals scored and conceded
        for team_code, team_df in df.groupby("team_code"):
            team_needs_processing = (df["team_code"] == team_code) & df[new_columns[0]].isna()

            if not team_needs_processing.any():
                continue

            fixture = (
                team_df
                .groupby(["season", "gameweek", "fixture"], as_index=False)
                .agg(goals_scored=("goals_scored", "sum"), goals_conceded=("goals_conceded", "max"))
                .sort_values(by=["season", "gameweek", "fixture"])
                .reset_index(drop=True)
            )

            fixture["roll_goals_scored"] = (
                fixture.goals_scored.shift(1).rolling(window=window, min_periods=1).sum().fillna(0)
            )

            fixture["roll_goals_conceded"] = (
                fixture.goals_conceded.shift(1).rolling(window=window, min_periods=1).sum().fillna(0)
            )

            team_map = {(r.season, r.gameweek, r.fixture): (r.roll_goals_scored, r.roll_goals_conceded)
                        for r in fixture.itertuples(index=False)}

            idxs = df.index[team_needs_processing]
            keys = list(zip(df.loc[idxs, "season"], df.loc[idxs, "gameweek"], df.loc[idxs, "fixture"]))
            vals = [team_map.get(k, (0, 0)) for k in keys]
            df.loc[idxs, f"team_goals_scored_rolling_{window}"]   = [int(v[0]) for v in vals]
            df.loc[idxs, f"team_goals_conceded_rolling_{window}"] = [int(v[1]) for v in vals]

        # Process opponent's goals scored and conceded
        for opponent_code, opponent_df in df.groupby("opponent_team_code"):
            opponent_needs_processing = (df["opponent_team_code"] == opponent_code) & df[new_columns[2]].isna()

            if not opponent_needs_processing.any():
                continue

            opp_team_df = df[df["team_code"] == opponent_code]
            if opp_team_df.empty:
                idxs = df.index[opponent_needs_processing]
                df.loc[idxs, f"opponent_goals_scored_rolling_{window}"] = 0
                df.loc[idxs, f"opponent_goals_conceded_rolling_{window}"] = 0
                continue
            
            fixture = (
                opp_team_df
                .groupby(["season", "gameweek", "fixture"], as_index=False)
                .agg(goals_scored=("goals_scored", "sum"), goals_conceded=("goals_conceded", "max"))
                .sort_values(by=["season", "gameweek", "fixture"])
                .reset_index(drop=True)
            )

            fixture["roll_goals_scored"] = (
                fixture.goals_scored.shift(1).rolling(window=window, min_periods=1).sum().fillna(0)
            )

            fixture["roll_goals_conceded"] = (
                fixture.goals_conceded.shift(1).rolling(window=window, min_periods=1).sum().fillna(0)
            )

            opp_map = {(r.season, r.gameweek, r.fixture): (r.roll_goals_scored, r.roll_goals_conceded)
                        for r in fixture.itertuples(index=False)}

            idxs = df.index[opponent_needs_processing]
            keys = list(zip(df.loc[idxs, "season"], df.loc[idxs, "gameweek"], df.loc[idxs, "fixture"]))
            vals = [opp_map.get(k, (0, 0)) for k in keys]
            df.loc[idxs, f"opponent_goals_scored_rolling_{window}"]   = [int(v[0]) for v in vals]
            df.loc[idxs, f"opponent_goals_conceded_rolling_{window}"] = [int(v[1]) for v in vals]

        return df

    def create_position_relevant_stats(self, df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
        """
        Transforms generic team stats into position-relevant stats
        
        Uses existing columns from add_team_goals_scored_conceded():
        - team_goals_scored_rolling_{window}
        - team_goals_conceded_rolling_{window}  
        - opponent_goals_scored_rolling_{window}
        - opponent_goals_conceded_rolling_{window}
        
        Creates position-specific columns:
        - position_team_stat_rolling_{window}
        - position_opponent_stat_rolling_{window}
        
        Logic:
        - FWD: team_goals, opponent_conceded (pure attack focus)
        - GK: team_conceded, opponent_goals (pure defense focus)
        - DEF: weighted 15% attack + 85% defense (receive some points for assists mainly)
        - MID: weighted 84% attack + 16% defense (reflects FPL scoring: 5pts goal vs 1pt clean sheet)
        """
        # Required input columns from your existing function
        required_columns = [
            f"team_goals_scored_rolling_{window}",
            f"team_goals_conceded_rolling_{window}",
            f"opponent_goals_scored_rolling_{window}",
            f"opponent_goals_conceded_rolling_{window}",
            "position"
        ]
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Please run add_team_goals_scored_conceded() first for window={window}.")
            return df

        df = df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)
        
        new_columns = [
            f"position_team_stat_rolling_{window}",
            f"position_opponent_stat_rolling_{window}",
        ]
        
        # Initialize new columns if they don't exist
        for col in new_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        # Check if there are any NaN values to process
        needs_processing = df[new_columns[0]].isna()
        if not needs_processing.any():
            return df
        
        # Forward: Attack focus
        forward_mask = needs_processing & (df.position.isin(["Forward", "FWD"]))
        if forward_mask.any():
            df.loc[forward_mask, f"position_team_stat_rolling_{window}"] = df.loc[forward_mask, f"team_goals_scored_rolling_{window}"].astype("float64")
            df.loc[forward_mask, f"position_opponent_stat_rolling_{window}"] = df.loc[forward_mask, f"opponent_goals_conceded_rolling_{window}"].astype("float64")

        # Defender: Weighted combination (15% attack + 85% defence)
        defender_mask = needs_processing & (df.position.isin(["Defender", "DEF"]))
        if defender_mask.any():
            df.loc[defender_mask, f"position_team_stat_rolling_{window}"] = (
                (df.loc[defender_mask, f"team_goals_scored_rolling_{window}"] * 0.15) +
                (df.loc[defender_mask, f"team_goals_conceded_rolling_{window}"] * 0.85)
            ).astype("float64")

            df.loc[defender_mask, f"position_opponent_stat_rolling_{window}"] = (
                (df.loc[defender_mask, f"opponent_goals_scored_rolling_{window}"] * 0.85) +
                (df.loc[defender_mask, f"opponent_goals_conceded_rolling_{window}"] * 0.15)
            ).astype("float64")

        # Goalkeeper: Defence focus
        goalkeeper_mask = needs_processing & df.position.isin(["Goalkeeper", "GK"])
        if goalkeeper_mask.any():
            df.loc[goalkeeper_mask, f"position_team_stat_rolling_{window}"] = df.loc[
                goalkeeper_mask, f"team_goals_conceded_rolling_{window}"
            ].astype("float64")

            df.loc[goalkeeper_mask, f"position_opponent_stat_rolling_{window}"] = df.loc[
                goalkeeper_mask, f"opponent_goals_scored_rolling_{window}"
            ].astype("float64")

        # Midfielder: Weighted combination (84% attack + 16% defence)
        midfielder_mask = needs_processing & (df.position.isin(["Midfielder", "MID"]))
        if midfielder_mask.any():
            df.loc[midfielder_mask, f"position_team_stat_rolling_{window}"] = (
                (df.loc[midfielder_mask, f"team_goals_scored_rolling_{window}"] * 0.84) +
                (df.loc[midfielder_mask, f"team_goals_conceded_rolling_{window}"] * 0.16)
            ).astype("float64")

            df.loc[midfielder_mask, f"position_opponent_stat_rolling_{window}"] = (
                (df.loc[midfielder_mask, f"opponent_goals_scored_rolling_{window}"] * 0.16) +
                (df.loc[midfielder_mask, f"opponent_goals_conceded_rolling_{window}"] * 0.84)
            ).astype("float64")
        
        unknown_positions = needs_processing & ~df.position.isin(["Forward", "FWD", "Defender", "DEF", "Goalkeeper", "GK", "Midfielder", "MID"])
        if unknown_positions.any():
            raise ValueError(f"Unknown positions found in dataset."
                             f"List of existent positions: {df.position.unique().tolist()}")

        return df

    def scale_features(self, df: pd.DataFrame, columns_to_scale: list = None) -> pd.DataFrame:
        """
        Scales selected numerical features using MinMaxScaler per season for each column independently
        """
        if not columns_to_scale:
            print("No columns selected for scaling.")
            return df
        
        # Sort data to ensure consistent processing
        df = df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)
        
        # Initialize scaled columns if they don't exist
        scaled_columns = [f"{col}_scaled" for col in columns_to_scale]
        for scaled_col in scaled_columns:
            if scaled_col not in df.columns:
                df[scaled_col] = np.nan
        
        needs_processing = df[scaled_columns[0]].isna()
        if not needs_processing.any():
            return df

        seasons = sorted(df["season"].unique())
        
        # Process each season independently
        for season in seasons:
            season_mask = df["season"] == season
            season_needs_processing = needs_processing & season_mask
            
            if not season_needs_processing.any():
                continue
            
            # Get all data for this season
            season_data = df[season_mask]
            
            # Scale each column independently for this season
            for col in columns_to_scale:
                if col not in df.columns:
                    print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                    continue
                    
                scaled_col = f"{col}_scaled"
                
                # Get season data for this column
                season_col_data = season_data[col].dropna()
                
                if len(season_col_data) == 0:
                    print(f"No valid data for column '{col}' in season {season}. Skipping.")
                    continue
                
                scaler = MinMaxScaler()
                
                if len(season_col_data) == 1:
                    # It will assign 0.0 to all values when min == max
                    scaler.fit(season_col_data.values.reshape(-1, 1))
                else:
                    scaler.fit(season_col_data.values.reshape(-1, 1))
                
                # Transform only the rows that need processing in this season
                rows_to_transform = df.loc[season_needs_processing, col]

                valid_mask = ~rows_to_transform.isna()
                if not valid_mask.any():
                    continue
                
                valid_values = rows_to_transform[valid_mask].values.reshape(-1, 1)
                scaled_values = scaler.transform(valid_values).flatten()
                
                valid_indices = rows_to_transform[valid_mask].index
                df.loc[valid_indices, scaled_col] = scaled_values.astype("float64")

        return df

    def assign_player_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns player importance ranking and contribution percentage for each player within their team per season
        """
        df = df.sort_values(by=["season", "team_code", "gameweek"]).reset_index(drop=True)

        importance_columns = ["team_total_points", "importance_rank", "importance", "cumulative_points"]
        for col in importance_columns:
            if col not in df.columns:
                df[col] = np.nan

        needs_processing = df["importance_rank"].isna()
        if not needs_processing.any():
            return df
        
        df_to_process = df[needs_processing].copy()

        # Calculate cumulative statistics
        df_to_process["team_total_points"] = (
            df_to_process.groupby(["season", "team_code"])["total_points"].shift(1).cumsum().fillna(0).astype("int64")
        )

        df_to_process["cumulative_points"] = (
            df_to_process.groupby(["season", "team_code", "code"])["total_points"].shift(1).cumsum().fillna(0).astype("int64")
        )

        # Calculate importance percentage
        df_to_process["importance"] = (
            (df_to_process["cumulative_points"] / df_to_process["team_total_points"].replace(0, 1)) * 100 
        ).fillna(0.0).astype("float64")

        # Rank players within team for each gameweek separately
        df_to_process["importance_rank"] = np.nan
        
        for season in df_to_process.season.unique():
            for team in df_to_process[df_to_process.season == season].team_code.unique():
                for gw in df_to_process[(df_to_process.season == season) & (df_to_process.team_code == team)].gameweek.unique():
                    
                    # Get all players in this team/season/gameweek
                    gw_mask = (
                        (df_to_process.season == season) & 
                        (df_to_process.team_code == team) & 
                        (df_to_process.gameweek == gw)
                    )
                    
                    gw_data = df_to_process[gw_mask]
                    
                    if len(gw_data) > 0:
                        # Rank players within this specific gameweek
                        ranks = gw_data["cumulative_points"].rank(method="dense", ascending=False).astype("int32")
                        df_to_process.loc[gw_mask, "importance_rank"] = ranks
        
        for col in importance_columns:
            df.loc[needs_processing, col] = df_to_process[col].values

        # Handle first gameweeks
        seasons = sorted(df.season.unique())
        first_gw_indices = []

        for player_id in df.code.unique():
            player_data = df[df.code == player_id]
            player_seasons = sorted(player_data.season.unique())

            for season in player_seasons:
                season_idx = seasons.index(season)
                player_season_data = player_data[player_data.season == season]
                first_gw = player_season_data.gameweek.min()
                first_gw_mask = (df.code == player_id) & (df.season == season) & (df.gameweek == first_gw)

                first_gw_idx = player_season_data[player_season_data.gameweek == first_gw].index
                first_gw_indices.extend(first_gw_idx.to_list())

                # Case 1: player's first season
                if season == player_seasons[0]:
                    current_gw_data = player_season_data[player_season_data.gameweek == first_gw]
                    if not current_gw_data.empty:
                        team_id = current_gw_data.team_code.iloc[0]

                        team_gw_data = df[
                            (df.season == season) &
                            (df.team_code == team_id) &
                            (df.gameweek == first_gw)
                        ]

                        player_points = current_gw_data.total_points.iloc[0]
                        team_total = team_gw_data.total_points.sum()

                        df.loc[first_gw_mask, "cumulative_points"] = player_points
                        df.loc[first_gw_mask, "team_total_points"] = team_total

                # Case 2: player's subsequent seasons
                else:
                    previous_season_data = None
                    previous_season = None

                    for i in range(season_idx - 1, -1, -1):
                        temp_season = seasons[i]
                        temp_data = player_data[player_data.season == temp_season]
                        if not temp_data.empty:
                            previous_season_data = temp_data
                            previous_season = temp_season
                            break

                    if previous_season_data is not None:
                        team_id = previous_season_data["team_code"].iloc[0]
                        player_prev_total = previous_season_data.total_points.sum()

                        team_prev_data = df[
                            (df.season == previous_season) &
                            (df.team_code == team_id)
                        ]

                        if not team_prev_data.empty:
                            team_prev_total = team_prev_data.groupby("code")["total_points"].sum().sum()

                        df.loc[first_gw_mask, "cumulative_points"] = player_prev_total
                        df.loc[first_gw_mask, "team_total_points"] = team_prev_total

        # Calculate importance and rank for first gameweeks
        if first_gw_indices:
            first_gw_data = df.loc[first_gw_indices].copy()

            first_gw_data["importance"] = (
                (first_gw_data["cumulative_points"] / first_gw_data["team_total_points"].replace(0, 1)) * 100
            ).fillna(0.0).astype("float64")

            first_gw_data["importance_rank"] = (
                first_gw_data.groupby(["season", "team_code"])["cumulative_points"]
                .rank(method="dense", ascending=False).astype("int32")
            )

            df.loc[first_gw_indices, "importance"] = first_gw_data["importance"].values.astype("float64")
            df.loc[first_gw_indices, "importance_rank"] = first_gw_data["importance_rank"].values.astype("int32")
            df.loc[first_gw_indices, "team_total_points"] = np.int64(0)
            df.loc[first_gw_indices, "cumulative_points"] = np.int64(0)

        return df
    
    def run_pipeline(
            self,
            df: pd.DataFrame,
            window: int = 8,
            diff_window: int = 5,
            team_window: int = 5,
            form_window: int = 4,
            columns_to_scale: list[str] | None = None,
            save_filename: str = "training_dataset.csv",
    ) -> pd.DataFrame:
        """
        Run the complete processing pipeline

        Parameters:
            df: Input dataframe to process
            window: Window for rolling averages
            diff_window: Window for difficulty-based averages
            team_window: Window for team goals calculations
            form_window: Window for form and minutes calculations
            columns_to_scale: list of columns to scale (optional)
            save_filename: Filename to save the processed dataset
        
        Returns:
            df: Processed dataframe with engineered features
        """
        # Specify the season period
        df["is_late_season"] = (df["gameweek"] >= 22).astype("Int8")

        # Add team goals scored/ conceded
        df = self.add_team_goals_scored_conceded(df, window=team_window)

        # Make the team stats position based
        df = self.create_position_relevant_stats(df, window=team_window)

        # Assign player importance
        df = self.assign_player_importance(df)
        df["importance_rank"] = df["importance_rank"].astype("int32")
        df["team_total_points"] = df["team_total_points"].astype("int64")
        df["cumulative_points"] = df["cumulative_points"].astype("int64")

        # Calculate form and average minutes (gor adjustent predictions)
        df = self.calculate_form_and_average_minutes(df, window=form_window)

        # Calculate rolling averages
        df = self.calculate_rolling_averages(df, window=window)

        # Calculate difficulty-based averages
        df = self.calculate_averages_difficulty(df, window=diff_window)

        # Scale features (is specified)
        if columns_to_scale:
            df = self.scale_features(df, columns_to_scale=columns_to_scale)

        if self.save_data and self.path:
            if save_filename:
                filename = save_filename
            else:
                filename = "training_dataset.csv"
            
            filepath = os.path.join(self.path, filename)
            df.to_csv(filepath, index=False)

        return df


class FPLFutureFeatureEngineering:
    """
    FPL data preprocessing class that handles feature engineering for the future fixtures dataset
    Uses past_df which has historical data of players for the calculations
    """
    def __init__(self, path: str = "", save_data: bool = True):
        """
        Parameters:
            path: Path for data storage
            save_data: Whether to save processed data to files
        """
        self.path = path
        self.save_data = save_data

    def calculate_averages_difficulty(self, past_df: pd.DataFrame, future_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Calculate difficulty-adjusted rolling averages for future fixtures with weighted combination of current-past season data
        """
        # Sort by player, season, and gameweek to ensure correct rolling calculation
        past_df = past_df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)
        future_df = future_df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)

        numeric_cols = [
            "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", 
            "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", 
            "red_cards", "saves", "bonus", "bps", "total_points", "threat", "ict_index", 
            "influence", "creativity", "expected_goals", "expected_assists", 
            "expected_goal_involvements", "expected_goals_conceded",
        ]
        
        for col in numeric_cols:
            future_df[f"{col}_average_difficulty_{window}"] = np.nan

        for idx in future_df.index:
            player_id = future_df.loc[idx, "code"]
            current_season = future_df.loc[idx, "season"]
            opponent_difficulty = future_df.loc[idx, "opponent_difficulty"]

            player_past_df = past_df[past_df.code == player_id].copy()
            
            # Case 1: New Premier League player
            if player_past_df.empty:
                for col in numeric_cols:
                    future_df.loc[idx, f"{col}_average_difficulty_{window}"] = 0.0
                continue
            
            # Case 2: Existing Premier League player
            else:
                seasons = sorted(player_past_df.season.unique())

                # Case 2A: New season
                if current_season not in seasons:
                    same_diff_df = player_past_df[
                        (player_past_df.season == player_past_df.season.max()) &
                        (player_past_df.opponent_difficulty == opponent_difficulty)
                    ]
                    # Case 2A1: same difficulty fixtures exist
                    if not same_diff_df.empty:
                        for col in numeric_cols:
                            future_df.loc[idx, f"{col}_average_difficulty_{window}"] = float(same_diff_df[col].tail(window).mean())
                    # Case 2A2: no same difficulty fixtures - use rolling averages of most recent gameweeks
                    else:
                        for col in numeric_cols:
                            future_df.loc[idx, f"{col}_average_difficulty_{window}"] = float(player_past_df[col].tail(window).mean())
                    
                # Case 2B: After first gameweek of the season
                else:
                    same_diff_df = player_past_df[
                        (player_past_df.season == current_season) &
                        (player_past_df.opponent_difficulty == opponent_difficulty)
                    ]
                    # Case 2B1: sufficient same difficulty fixtures data
                    if len(same_diff_df) >= window:
                        for col in numeric_cols:
                            future_df.loc[idx, f"{col}_average_difficulty_{window}"] = float(same_diff_df[col].tail(window).mean())
                    # Case 2B2: insufficient same difficulty fixtures data for current season
                    else:
                        n_current = len(same_diff_df)
                        weight_current = n_current / window
                        past_data_needed = window - n_current

                        if len(seasons) > 1:
                            past_season = seasons[-2]

                            previous_season_same_diff_df = player_past_df[
                                (player_past_df.season == past_season) &
                                (player_past_df.opponent_difficulty == opponent_difficulty)
                            ]
                        else:
                            previous_season_same_diff_df = pd.DataFrame()

                        if previous_season_same_diff_df.empty and not same_diff_df.empty:
                            # No same difficulty in previous season - use only data from current season
                            for col in numeric_cols:
                                future_df.loc[idx, f"{col}_average_difficulty_{window}"] = float(same_diff_df[col].mean())
                        elif not previous_season_same_diff_df.empty:
                            # Existing same difficulty data in previous season
                            for col in numeric_cols:
                                current_avg = same_diff_df[col].mean() if not same_diff_df.empty else 0.0
                                past_avg = previous_season_same_diff_df[col].tail(past_data_needed).mean()
                                future_df.loc[idx, f"{col}_average_difficulty_{window}"] = float(
                                    weight_current * current_avg + (1.0 - weight_current) * past_avg
                                )
                        else:
                            # No same difficulty data in either current or previous season - use general past data
                            current_df = player_past_df[player_past_df.season == current_season]

                            if len(current_df) >= window:
                                # Sufficient general data in current season
                                for col in numeric_cols:
                                    future_df.loc[idx, f"{col}_average_difficulty_{window}"] = float(current_df[col].tail(window).mean())
                            else:
                                # Insufficient data in current season - use general past data
                                n_current = len(current_df)
                                weight_current = n_current / window
                                past_data_needed = window - n_current

                                for col in numeric_cols:
                                    current_avg = current_df[col].mean() 
                                    past_avg = player_past_df[col].tail(past_data_needed).mean()
                                    future_df.loc[idx, f"{col}_average_difficulty_{window}"] = float(
                                        weight_current * current_avg + (1.0 - weight_current) * past_avg
                                    )      

        return future_df

    def calculate_rolling_averages(self, past_df: pd.DataFrame, future_df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
        """
        Calculate rolling averages for future fixtures with weighted combination of current-past season data
        """
        # Sort by player, season, and gameweek to ensure correct rolling calculation
        past_df = past_df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)
        future_df = future_df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)

        numeric_cols = [
            "minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded", 
            "own_goals", "penalties_saved", "penalties_missed", "yellow_cards", 
            "red_cards", "saves", "bonus", "bps", "total_points", "threat", "ict_index", 
            "influence", "creativity", "expected_goals", "expected_assists", 
            "expected_goal_involvements", "expected_goals_conceded",
        ]
        
        for col in numeric_cols:
            future_df[f"{col}_average_rolling_{window}"] = np.nan

        for idx in future_df.index:
            player_id = future_df.loc[idx, "code"]
            current_season = future_df.loc[idx, "season"]

            player_past_df = past_df[past_df.code == player_id].copy()

            # Case 1: New Premier League player
            if player_past_df.empty:
                for col in numeric_cols:
                    future_df.loc[idx, f"{col}_average_rolling_{window}"] = 0.0
                continue

            # Case 2: Existing Premier League player
            else:
                seasons = sorted(player_past_df.season.unique())

                # Case 2A: New season
                if current_season not in seasons:
                    for col in numeric_cols:
                        future_df.loc[idx, f"{col}_average_rolling_{window}"] = float(player_past_df[col].tail(window).mean())
                
                # Case 2B: After first gameweek of the season
                else:
                    same_season_df = player_past_df[player_past_df.season == current_season]
                    # Case 2B1: sufficient same season data
                    if len(same_season_df) >= window:
                        for col in numeric_cols:
                            future_df.loc[idx, f"{col}_average_rolling_{window}"] = float(same_season_df[col].tail(window).mean())
                    # Case 2B2: insufficient same season data
                    else:
                        n_current = len(same_season_df)
                        weight_current = n_current / window
                        past_data_needed = window - n_current

                        if len(seasons) > 1:
                            past_season = seasons[-2]
                            previous_season_df = player_past_df[player_past_df.season == past_season]
                        else:
                            previous_season_df = pd.DataFrame()
                    
                        if previous_season_df.empty:
                            # No previous season data - use only current season data
                            for col in numeric_cols:
                                future_df.loc[idx, f"{col}_average_rolling_{window}"] = float(same_season_df[col].mean())
                        else:
                            # Existing previous season data
                            for col in numeric_cols:
                                current_avg = same_season_df[col].mean()
                                past_avg = previous_season_df[col].tail(past_data_needed).mean()
                                future_df.loc[idx, f"{col}_average_rolling_{window}"] = float(
                                    weight_current * current_avg + (1.0 - weight_current) * past_avg
                                )

        return future_df

    def calculate_form_and_average_minutes(self, past_df: pd.DataFrame, future_df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
        """
        Calculate rolling averages for form (total_points) and minutes with normalization
        """
        # Sort data to ensure correct rolling calculation
        past_df = past_df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)
        future_df = future_df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)

        new_columns = [
            f"form_average_rolling_{window}",
            f"form_average_rolling_{window}_normalized",
            f"form_average_rolling_{window}_sqrt_normalized", 
            f"minutes_average_rolling_{window}",
            f"minutes_average_rolling_{window}_normalized",
        ]

        for col in new_columns:
            future_df[col] = np.nan

        for player_id in future_df.code.unique():
            player_past_df = past_df[past_df.code == player_id].copy()

            # Case 1: New Premier League player
            if player_past_df.empty:
                for col in new_columns:
                    future_df.loc[future_df.code == player_id, col] = 0.0
                continue

            # Case 2: Existing Premier League player
            player_season_df = player_past_df[player_past_df.season == player_past_df.season.max()]

            future_df.loc[future_df.code == player_id, f"form_average_rolling_{window}"] = float(player_season_df.total_points.tail(window).mean())
            future_df.loc[future_df.code == player_id, f"minutes_average_rolling_{window}"] = float(player_season_df.minutes.tail(window).mean())

        # Calculate normalized form and minutes
        form_max = future_df[f"form_average_rolling_{window}"].max()
        if form_max == 0 or pd.isna(form_max):
            form_max = 0.1 # Avoid division by zero

        form_sqrt_max = np.sqrt(np.maximum(future_df[f"form_average_rolling_{window}"], 0)).max()
        if form_sqrt_max == 0 or pd.isna(form_sqrt_max):
            form_sqrt_max = 0.1 # Avoid division by zero

        future_df[f"form_average_rolling_{window}_normalized"] = (
            future_df[f"form_average_rolling_{window}"] / form_max
        ).astype("float64")

        future_df[f"form_average_rolling_{window}_sqrt_normalized"] = (
            np.sqrt(np.maximum(future_df[f"form_average_rolling_{window}"], 0)) / form_sqrt_max
        ).astype("float64")

        future_df[f"minutes_average_rolling_{window}_normalized"] = (
            future_df[f"minutes_average_rolling_{window}"] / 90
        ).astype("float64")
        
        return future_df

    def add_team_goals_scored_conceded(self, past_df: pd.DataFrame, future_df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
        """
        Add rolling goals scored and conceded by the player's team and its opponent
        """

        new_columns = [
            f"team_goals_scored_rolling_{window}",
            f"team_goals_conceded_rolling_{window}",
            f"opponent_goals_scored_rolling_{window}",
            f"opponent_goals_conceded_rolling_{window}",
        ]

        past_df = past_df.sort_values(by=["season", "team_code", "gameweek"]).reset_index(drop=True)
        future_df = future_df.sort_values(by=["season", "gameweek", "team_code"]).reset_index(drop=True)

        for col in new_columns:
            future_df[col] = np.nan

        # Process team's goals scored and conceded
        for team_id in future_df.team_code.unique():
            team_df = past_df[past_df.team_code == team_id]

            if team_df.empty:
                future_df.loc[future_df.team_code == team_id, f"team_goals_scored_rolling_{window}"] = 0
                future_df.loc[future_df.team_code == team_id, f"team_goals_conceded_rolling_{window}"] = 0
            
            else:
                last_keys = (
                    team_df[["season", "gameweek", "fixture"]]
                    .drop_duplicates()
                    .sort_values(by=["season", "gameweek", "fixture"])
                    .tail(window)
                )
                team_last = team_df.merge(last_keys, on=["season", "gameweek", "fixture"], how="inner")

                team_last["goals_scored"] = team_last["goals_scored"].fillna(0)
                team_last["goals_conceded"] = team_last["goals_conceded"].fillna(0)

                scored_per_gw = (
                    team_last.groupby(["season", "gameweek", "fixture"])["goals_scored"].sum().reset_index()
                )

                conceded_per_gw = (
                    team_last.groupby(["season", "gameweek", "fixture"])["goals_conceded"].max().reset_index()
                )

                team_scored = int(scored_per_gw.goals_scored.sum())
                team_conceded = int(conceded_per_gw.goals_conceded.sum())

                future_df.loc[future_df.team_code == team_id, f"team_goals_scored_rolling_{window}"] = team_scored
                future_df.loc[future_df.team_code == team_id, f"team_goals_conceded_rolling_{window}"] = team_conceded
        
        for opponent_id in future_df.opponent_team_code.unique():
            opponent_df = past_df[past_df.team_code == opponent_id]

            if opponent_df.empty:
                future_df.loc[future_df.opponent_team_code == opponent_id, f"opponent_goals_scored_rolling_{window}"] = 0
                future_df.loc[future_df.opponent_team_code == opponent_id, f"opponent_goals_conceded_rolling_{window}"] = 0

            else:
                last_keys = (
                    opponent_df[["season", "gameweek", "fixture"]]
                    .drop_duplicates()
                    .sort_values(by=["season", "gameweek", "fixture"])
                    .tail(window)
                )
                opponent_last = opponent_df.merge(last_keys, on=["season", "gameweek", "fixture"], how="inner")

                opponent_last["goals_scored"] = opponent_last["goals_scored"].fillna(0)
                opponent_last["goals_conceded"] = opponent_last["goals_conceded"].fillna(0)

                scored_per_gw = (
                    opponent_last.groupby(["season", "gameweek", "fixture"])["goals_scored"].sum().reset_index()
                )

                conceded_per_gw = (
                    opponent_last.groupby(["season", "gameweek", "fixture"])["goals_conceded"].max().reset_index()
                )

                opponent_scored = int(scored_per_gw.goals_scored.sum())
                opponent_conceded = int(conceded_per_gw.goals_conceded.sum())

                future_df.loc[future_df.opponent_team_code == opponent_id, f"opponent_goals_scored_rolling_{window}"] = opponent_scored
                future_df.loc[future_df.opponent_team_code == opponent_id, f"opponent_goals_conceded_rolling_{window}"] = opponent_conceded
        
        return future_df

    def create_position_relevant_stats(self, df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
        """
            Transforms generic team stats into position-relevant stats using existing columns
            
            Uses existing columns from add_team_goals_scored_conceded():
            - team_goals_scored_rolling_{window}
            - team_goals_conceded_rolling_{window}  
            - opponent_goals_scored_rolling_{window}
            - opponent_goals_conceded_rolling_{window}
            
            Creates position-specific columns:
            - position_team_stat_rolling_{window}
            - position_opponent_stat_rolling_{window}
            
            Logic:
            - FWD: team_goals, opponent_conceded (pure attack focus)
            - GK: team_conceded, opponent_goals (pure defense focus)
            - DEF: weighted 15% attack + 85% defense (receive some points for assists mainly)
            - MID: weighted 84% attack + 16% defense (reflects FPL scoring: 5pts goal vs 1pt clean sheet)
            """
        necess_columns = [
            f"team_goals_scored_rolling_{window}",
            f"team_goals_conceded_rolling_{window}",
            f"opponent_goals_scored_rolling_{window}",
            f"opponent_goals_conceded_rolling_{window}",
            "position"
        ]

        new_columns = [
            f"position_team_stat_rolling_{window}",
            f"position_opponent_stat_rolling_{window}"
        ]

        missing_columns = [col for col in necess_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Run add_team_goals_scored+conceded() first for window={window}")
            return df
        
        for col in new_columns:
            df[col] = np.nan
        
        df = df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)

        # Forwards
        forward_mask = df.position.isin(["Forward", "FWD"])
        df.loc[forward_mask, f"position_team_stat_rolling_{window}"] = df.loc[
            forward_mask, f"team_goals_scored_rolling_{window}"
        ].astype("float64")

        df.loc[forward_mask, f"position_opponent_stat_rolling_{window}"] = df.loc[
            forward_mask, f"opponent_goals_conceded_rolling_{window}"
        ].astype("float64")
        
        # Defenders
        defender_mask = df.position.isin(["Defender", "DEF"])
        df.loc[defender_mask, f"position_team_stat_rolling_{window}"] = (
            (df.loc[defender_mask, f"team_goals_scored_rolling_{window}"] * 0.15) +
            (df.loc[defender_mask, f"team_goals_conceded_rolling_{window}"] * 0.85)
        ).astype("float64")

        df.loc[defender_mask, f"position_opponent_stat_rolling_{window}"] = (
            (df.loc[defender_mask, f"opponent_goals_scored_rolling_{window}"] * 0.85) +
            (df.loc[defender_mask, f"opponent_goals_conceded_rolling_{window}"] * 0.15)
        ).astype("float64")

        # Goalkeepers
        goalkeeper_mask = df.position.isin(["Goalkeeper", "GK"])
        df.loc[goalkeeper_mask, f"position_team_stat_rolling_{window}"] = df.loc[
            goalkeeper_mask, f"team_goals_conceded_rolling_{window}"
        ].astype("float64")

        df.loc[goalkeeper_mask, f"position_opponent_stat_rolling_{window}"] = df.loc[
            goalkeeper_mask, f"opponent_goals_scored_rolling_{window}"
        ].astype("float64")
        
        # Midfielders
        midfielder_mask = df.position.isin(["Midfielder", "MID"])
        df.loc[midfielder_mask, f"position_team_stat_rolling_{window}"] = (
            (df.loc[midfielder_mask, f"team_goals_scored_rolling_{window}"] * 0.84) +
            (df.loc[midfielder_mask, f"team_goals_conceded_rolling_{window}"] * 0.16)
        ).astype("float64")

        df.loc[midfielder_mask, f"position_opponent_stat_rolling_{window}"] = (
            (df.loc[midfielder_mask, f"opponent_goals_scored_rolling_{window}"] * 0.16) +
            (df.loc[midfielder_mask, f"opponent_goals_conceded_rolling_{window}"] * 0.84)
        ).astype("float64")
        
        return df

    def scale_features(self, past_df: pd.DataFrame, future_df: pd.DataFrame, columns_to_scale: list = None) -> pd.DataFrame:
        """
        Scales selected numerical features using MinMaxScaler per season for each column independently
        """
        if columns_to_scale is None:
            return future_df
        
        past_df = past_df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)
        future_df = future_df.sort_values(by=["code", "season", "gameweek"]).reset_index(drop=True)
        
        scaled_columns = [f"{col}_scaled" for col in columns_to_scale]
        for col in scaled_columns:
            future_df[col] = np.nan

        season_df = past_df[past_df.season == past_df.season.max()]
        
        for col in columns_to_scale:
            season_col_data = season_df[col].dropna()

            if len(season_col_data) == 0:
                print(f"Warning: No valid data for column '{col}'. Skipping.")
                continue

            scaler = MinMaxScaler()
            scaler.fit(season_col_data.values.reshape(-1, 1))

            future_df[f"{col}_scaled"] = scaler.transform(future_df[col].values.reshape(-1, 1)).flatten().astype("float64")
        
        return future_df

    def assign_player_importance(self, past_df: pd.DataFrame, future_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns player importance ranking and contribution percentage for each player within their team per season
        """
        importance_columns = [
            "team_total_points",
            "importance_rank",
            "importance",
            "cumulative_points"
        ]

        for col in importance_columns:
            future_df[col] = np.nan
        
        past_df = past_df.sort_values(by=["season", "team_code", "gameweek"])
        future_df = future_df.sort_values(by=["season", "team_code", "gameweek"])
        
        # When we are in the first gameweek past_season_df will contain the precious season to calculate importances
        # in the next gameweeks it will contain data from current season
        past_season_df = past_df[past_df.season == past_df.season.max()]

        team_totals = past_season_df.groupby("team_code")["total_points"].sum()
        player_totals = past_season_df.groupby("code")["total_points"].sum()

        future_df["team_total_points"] = future_df["team_code"].map(team_totals).fillna(0).astype("int64")
        future_df["cumulative_points"] = future_df["code"].map(player_totals).fillna(0).astype("int64")
        
        future_df["importance"] = (
            (future_df["cumulative_points"] / future_df["team_total_points"].replace(0, 1)) * 100
        ).fillna(0.0).astype("float64")

        for team in future_df.team_code.unique():
            future_df.loc[future_df.team_code == team, "importance_rank"] = future_df.loc[
                future_df.team_code == team, "cumulative_points"].rank(
                method="dense", ascending=False
            ).astype("int32")

        seasons = sorted(past_df.season.unique())
        current_season = future_df.season.iloc[0]

        # If we are in the first gameweek reset the team, cumulative points to 0 for consistency
        if current_season not in seasons:
            future_df.team_total_points = 0
            future_df.cumulative_points = 0
        
        return future_df

    def run_pipeline(
            self,
            past_df: pd.DataFrame,
            future_df: pd.DataFrame,
            window: int = 8,
            diff_window: int = 5,
            team_window: int = 5,
            form_window: int = 4,
            columns_to_scale: list[str] | None = None,
            save_filename: str = "prediction_dataset.csv",
    ) -> pd.DataFrame:
        """
        Run the complete processing pipeline

        Parameters:
            past_df: Historical data of players for calculations
            future_df: Future fixtures for predictions
            window: Window for rolling averages
            diff_window: Window for difficulty-based averages
            team_window: Window for team stats calculations
            form_window: Window specifically for form and minutes
            columns_to_scale: list of columns to scale (optional)
            save_filename: Filename to save the processed dataset

        Returns:
            future_df: Processed dataframe with engineered features for future fixtures
        """
        future_df["is_late_season"] = (future_df.gameweek >= 22).astype("Int8")

        future_df = self.add_team_goals_scored_conceded(past_df, future_df, window=team_window)

        future_df = self.create_position_relevant_stats(future_df, window=team_window)

        future_df = self.assign_player_importance(past_df, future_df)
        future_df["importance_rank"] = future_df["importance_rank"].astype("int32")
        future_df["team_total_points"] = future_df["team_total_points"].astype("int64")
        future_df["cumulative_points"] = future_df["cumulative_points"].astype("int64")

        future_df = self.calculate_form_and_average_minutes(past_df, future_df, window=form_window)

        future_df = self.calculate_rolling_averages(past_df, future_df, window=window)

        future_df = self.calculate_averages_difficulty(past_df, future_df, window=diff_window)

        if columns_to_scale:
            future_df = self.scale_features(past_df, future_df, columns_to_scale=columns_to_scale)

        if self.save_data and self.path:
            if save_filename:
                filename = save_filename
            else:
                filename = "prediction_dataset.csv"
            
            filepath = os.path.join(self.path, filename)
            future_df.to_csv(filepath, index=False)

        return future_df
