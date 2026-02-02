"""
FPL Model Utilities - Shared data transformation and utility functions for ML models
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import os
from sklearn.preprocessing import LabelEncoder
from typing import Any

def filter_top_players(df: pd.DataFrame, top_pct: float = 0.7, min_minutes_pct: float = 0.2) -> pd.DataFrame:
    """
    Filter top performing players per season and position based on total points

    Parameters:
        df: DataFrame with player gameweek data
        top_pct: fraction of players to filter out (0.7 keeps top 30%)
        min_minutes_pct: minimum fraction of season minutes required for player inclusion (0.2 keeps players with at least 20% of possible minutes)
    """
    # Calculate total points and games played per player-season
    player_stats = df.groupby(["code", "season", "position"]).agg(
        total_points_season=("total_points", "sum"),
        total_minutes_played=("minutes", "sum")
    ).reset_index()

    season_gw_counts = df.groupby("season")["gameweek"].nunique().to_dict()
    player_stats["max_possible_minutes"] = player_stats["season"].map(
        lambda s: season_gw_counts[s] * 90
    )

    # Set minimum minutes threshold
    player_stats["min_minutes_required"] = player_stats["max_possible_minutes"] * min_minutes_pct
    qualified_players = player_stats[player_stats["total_minutes_played"] >= player_stats["min_minutes_required"]]

    # Filter top X% per season and position
    top_players_list = []
    
    for (season, position), group in qualified_players.groupby(["season", "position"]):
        threshold = group["total_points_season"].quantile(top_pct)
        top_group = group[group["total_points_season"] >= threshold].copy()
        top_players_list.append(top_group)
    
    top_players = pd.concat(top_players_list, ignore_index=True)

    keep_set = set(zip(top_players["code"], top_players["season"]))

    return df[df.apply(lambda row: (row["code"], row["season"]) in keep_set, axis=1)]

def split_and_duplicate_averages(
        data: pd.DataFrame,
        rolling_cols: list[str],
        average_cols: list[str],
        include_test_set: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation(/test) sets with proper feature duplication to prevent data leakage
    
    Parameters:
        data: input DataFrame with player gameweek data
        rolling_cols: list of rolling average column names to duplicate
        average_cols: list of difficulty-based average column names to fill
        include_test_set: if True, split last 10 GWs (5 val + 5 test). If False, use last 5 GWs for validation only.
    
    Returns:
        If include_test_set=True: (train_data, validation_data, test_data)
        If include_test_set=False: (train_data, validation_data)
    """
    train_sets = []
    validation_sets = []
    test_sets = [] if include_test_set else None

    # Determine split parameters based on test set inclusion
    min_gameweeks = 10 if include_test_set else 5
    
    for _, player_data in data.groupby("code"):
        player_data = player_data.sort_values("gameweek")

        # Check if player has enough gameweeks
        if len(player_data) < min_gameweeks:
            train_sets.append(player_data)
            continue

        if include_test_set:
            # Last 10 GWs split into val (5) + test (5)
            last_gws = player_data.iloc[-10:].copy()
            validation_set = last_gws.iloc[:5].copy()
            test_set = last_gws.iloc[5:].copy()
            train_set = player_data.iloc[:-10].copy()
        else:
            # Last 5 GWs for validation only
            validation_set = player_data.iloc[-5:].copy()
            test_set = None
            train_set = player_data.iloc[:-5].copy()

        for col in rolling_cols:
            if len(validation_set) > 0 and col in validation_set.columns:
                validation_set[col] = validation_set[col].iloc[0]   # Broadcast first value to all rows
            
            if include_test_set and test_set is not None and col in test_set.columns:
                test_set[col] = test_set[col].iloc[0]               # Broadcast first value to all rows

        # Fill validation set difficulty averages
        for idx in validation_set.index[1:]:
            row = validation_set.loc[idx]
            difficulty = row["opponent_difficulty"]
            first_val_row = validation_set.iloc[0]

            if difficulty == first_val_row["opponent_difficulty"]:
                for col in average_cols:
                    validation_set.at[idx, col] = first_val_row[col]
            else:
                last_train_row = train_set[train_set["opponent_difficulty"] == difficulty].iloc[-1:]
                if not last_train_row.empty:
                    for col in average_cols:
                        validation_set.at[idx, col] = float(last_train_row[col].iloc[0])

        # Fill test set difficulty averages
        if include_test_set and test_set is not None:
            for idx in test_set.index[1:]:
                row = test_set.loc[idx]
                difficulty = row["opponent_difficulty"]
                first_test_row = test_set.iloc[0]

                if difficulty == first_test_row["opponent_difficulty"]:
                    for col in average_cols:
                        test_set.at[idx, col] = first_test_row[col]
                else:
                    # Look in validation set first, then train set
                    last_val_row = validation_set[validation_set["opponent_difficulty"] == difficulty].iloc[-1:]
                    if not last_val_row.empty:
                        for col in average_cols:
                            test_set.at[idx, col] = float(last_val_row[col].iloc[0])
                    else:
                        last_train_row = train_set[train_set["opponent_difficulty"] == difficulty].iloc[-1:]
                        if not last_train_row.empty:
                            for col in average_cols:
                                test_set.at[idx, col] = float(last_train_row[col].iloc[0])

        train_sets.append(train_set)
        validation_sets.append(validation_set)
        if include_test_set:
            test_sets.append(test_set)

    train_data = pd.concat(train_sets).reset_index(drop=True)
    validation_data = pd.concat(validation_sets).reset_index(drop=True)
    
    if include_test_set:
        test_data = pd.concat(test_sets).reset_index(drop=True)
        return train_data, validation_data, test_data
    else:
        return train_data, validation_data
    
def transform_features(
        data: pd.DataFrame,
        features: list[str], 
        categorical_features: list[str] | None = None,
        transformer: dict | None = None,
) -> tuple[pd.DataFrame, dict, list, list]:
    """
    Transform features by applying categorical encoding and feature reversing
    
    Parameters:
        data: input DataFrame
        features: list of feature names to use
        categorical_features: list of categorical features to encode
        transformer: existing transformer dict (for prediction)
        
    Returns:
        (transformed_data, transformer_dict, updated_features, active_categorical_features)
    """
    data_transformed = data.copy()

    # Detect if a transformer is provided
    for_training = transformer is None

    if for_training:
        transformer = {
            "label_encoders": {},
            "reversed_features": {},
            "max_values": {},
        }

    # Categorical encoding
    if for_training:
        for cat_feature in categorical_features:
            encoded_feature = f"{cat_feature}_encoded"

            # Check if we should encode this feature
            if (cat_feature in data_transformed.columns and
                (encoded_feature in features or cat_feature in features)):
                encoder = LabelEncoder()
                data_transformed[encoded_feature] = encoder.fit_transform(data_transformed[cat_feature].astype(str))
                transformer["label_encoders"][cat_feature] = encoder
    
    # Apply existing encoders
    else:
        for cat_feature, encoder in transformer["label_encoders"].items():
            encoded_feature = f"{cat_feature}_encoded"
            if cat_feature in data_transformed.columns:
                data_transformed[encoded_feature] = encoder.transform(data_transformed[cat_feature].astype(str))

    # Feature reversing
    features_to_reverse = [
        "opponent_difficulty",
        "importance_rank",
        "penalties_order",
        "corners_and_indirect_freekicks_order",
        "direct_freekicks_order",
    ]

    features_to_reverse = [f for f in features_to_reverse if f in features]
    if for_training:
        for feature in features_to_reverse:
            max_value = data_transformed[feature].max()
            reversed_feature = f"{feature}_reversed"

            data_transformed[reversed_feature] = max_value + 1 - data_transformed[feature]
            transformer["reversed_features"][feature] = reversed_feature
            transformer["max_values"][feature] = max_value
    
    else:
        # Apply existing reversed features
        for feature, reversed_feature in transformer["reversed_features"].items():
            max_value = transformer["max_values"][feature]
            data_transformed[reversed_feature] = max_value + 1 - data_transformed[feature]

    # Update feature list to use transformed features
    updated_features = []
    for feature in features:
        if feature in transformer["reversed_features"]:
            updated_features.append(transformer["reversed_features"][feature])
        else:
            updated_features.append(feature)

    # Get active categorical features after encoding
    active_categorical = []
    for cat_feature in categorical_features:
        encoded_feature = f"{cat_feature}_encoded"
        if encoded_feature in updated_features:
            active_categorical.append(encoded_feature)
        elif cat_feature in updated_features:
            active_categorical.append(cat_feature)

    return data_transformed, transformer, updated_features, active_categorical

def prepare_model_data(
        data: pd.DataFrame,
        features: list[str],
        rolling_cols: list[str],
        difficulty_cols: list[str],
        categorical_features: list[str] | None = None,
        include_test_set: bool = False,
        top_players_pct: float = 0.7,
        minutes_to_keep_pct: float = 0.2,
        target_column: str = "total_points",
        random_state: int = 42,
    ):
    """
    Complete data preparation pipeline for FPL model training
    
    Parameters:
        data: raw FPL DataFrame
        features: list of feature names
        rolling_cols: list of rolling average column names to duplicate
        difficulty_cols: list of difficulty-based average column names to fill
        categorical_features: list of categorical features to encode
        include_test_set: whether to create test set (last 10 GWs) or just validation (last 5 GWs)
        top_players_pct: threshold for mainting top players (0.7: keep 30% of top players, 0: keep all players)
        minutes_to_keep_pct: minimum fraction of season gameweeks required for player inclusion (0: keep all players)
        target_column: name of target column
        random_state: random state for reproducibility
        
    Returns:
        If include_test_set=True: (X_train, y_train), (X_val, y_val), (X_test, y_test), transformer, feature_names, active_categorical
        If include_test_set=False: (X_train, y_train), (X_val, y_val), transformer, feature_names, active_categorical
    """
    data = data.copy()
    
    # If we train with codes we should delete the players that do not exist in the current season
    if "code" in categorical_features:
        codes_map = data[data.season == data.season.max()].code.unique()
        data = data[data.code.isin(codes_map)]
        
    data = filter_top_players(data, top_pct=top_players_pct, min_minutes_pct=minutes_to_keep_pct)

    data_transformed, transformer, updated_features, active_categorical = transform_features(
        data=data,
        features=features,
        categorical_features=categorical_features
    )

    data_transformed = data_transformed.sample(frac=1, random_state=random_state).sort_values(
        by=["season", "gameweek"]
    ).reset_index(drop=True)
    
    if include_test_set:
        train_set, val_set, test_set = split_and_duplicate_averages(
            data_transformed, rolling_cols, difficulty_cols, include_test_set=True
        )
    else:
        train_set, val_set = split_and_duplicate_averages(
            data_transformed, rolling_cols, difficulty_cols, include_test_set=False
        )
        test_set = None

    X_train = train_set[updated_features]
    y_train = train_set[target_column]
    X_val = val_set[updated_features]
    y_val = val_set[target_column]

    if test_set is not None:
        X_test = test_set[updated_features]
        y_test = test_set[target_column]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test), transformer, updated_features, active_categorical, train_set, val_set, test_set
    
    else:
        return (X_train, y_train), (X_val, y_val), transformer, updated_features, active_categorical, train_set, val_set
    
def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    """
    Calculate standard regression metrics.
    
    Parameters:
        y_true: true target values
        y_pred: predicted values
        prefix: prefix for metric names (e.g., 'train_', 'val_')
        
    Returns:
        Dictionary with metric names and values
    """
    metrics = {
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}rmse": root_mean_squared_error(y_true, y_pred),
    }

    return metrics

def print_results(results: dict[str, Any]) -> None:
    for dataset in ["train", "val", "test"]:
        mae_key = f"{dataset}_mae"
        rmse_key = f"{dataset}_rmse"

        if mae_key in results and rmse_key in results:
            dataset_name = dataset.capitalize()
            print(f"{dataset_name:>2} MAE: {results[mae_key]:.4f} ||| RMSE: {results[rmse_key]:.4f}")

    if "num_features" in results:
        print(f"Features used: {results['num_features']}")

    if "train_mae" in results and "val_mae" in results:
        overfitting = results["val_mae"] - results["train_mae"]
        print(f"Overfitting: (val-train MAE): {overfitting:.4f}")

def save_results_to_csv(
        results: dict[str, Any],
        model_name: str,
        model_params: dict[str, Any],
        csv_path: str = "models_results.csv",
) -> None:
    base_row = {
        **model_params
    }

    rows_to_add = []

    # Row 1: Base results with hyperparameters
    base_model_row = {
        "model_name": model_name,
        **base_row,
        **results["base_metrics"],
        "overfitting": results['base_metrics']['val_mae'] - results['base_metrics']['train_mae'],
    }

    if "best_params" in results:
        for name, value in results["best_params"].items():
            base_model_row[f"hp_{name}"] = value

    rows_to_add.append(base_model_row)

    # Rows 2+: Adjusted results
    adjustment_types = ["minutes_adjusted", "form_adjusted", "form_sqrt_adjusted"]
    for adj_type in adjustment_types:
        if adj_type in results:
            adj_row = {
                "model_name": f"{model_name}_{adj_type}",
                **base_row,
                **results[adj_type],
                "overfitting": results[adj_type]['val_mae'] - results[adj_type]['train_mae'],
            }
            rows_to_add.append(adj_row)

    new_rows_df = pd.DataFrame(rows_to_add)

    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            
            # Check for existing models and update/add accordingly
            for _, row in new_rows_df.iterrows():
                model_name_check = row["model_name"]
                if model_name_check in existing_df["model_name"].values:
                    print(f"Model '{model_name_check}' already exists. Updating...")
                    mask = existing_df["model_name"] == model_name_check
                    for col in row.index:
                        if col in existing_df.columns:
                            existing_df.loc[mask, col] = row[col]
                        else:
                            existing_df[col] = np.nan
                            existing_df.loc[mask, col] = row[col]
                else:
                    new_row_df = pd.DataFrame([row])
                    existing_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            
            updated_df = existing_df
                    
        except Exception as e:
            print(f"Error reading existing file: {e}")
            print(f"Creating new file...")
            updated_df = new_rows_df
    else:
        updated_df = new_rows_df
    
    try:
        updated_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        
    return updated_df

def calculate_sample_weights(
        data: pd.DataFrame,
        alpha_recency: float = 0.1, 
        alpha_points: float = 1.0,
        alpha_combined: float = 0.5,
        rol_window: int = 8,
        target: str = "total_points",
) -> np.ndarray | None:
    """
    Calculate combined sample weights based on recency and performance
    
    Parameters:
        data: DataFrame with gameweek and points data
        alpha_recency: weight for recency (higher = more weight on recent data)
        alpha_points: weight for performance (higher = more weight on high-performing players)
        alpha_combined: balance between recency and performance weights
        rol_window: rolling window for performance calculation
        target: target column name
        
    Returns:
        Array of sample weights or None if no weighting
    """
    n_samples = len(data)

    # Case 1: both alphas are 0, there is no weighting
    if alpha_recency == 0 and alpha_points == 0:
        return None
    
    # Case 2: only recency weighting
    elif alpha_points == 0:
        weights_1 = np.exp(alpha_recency * np.arange(n_samples) / n_samples)
        weights_1 /= np.sum(weights_1)
        return weights_1
    
    # Case 3: only performance weighting
    elif alpha_recency == 0:
        performance_col = f"{target}_average_rolling_{rol_window}"
        points = data[performance_col]
        weights_2 = np.where(
            points >= 1,
            np.exp(alpha_points * points),
            np.maximum(1 + alpha_points * points, 0.1)
        )
        weights_2 /= np.sum(weights_2)
        return weights_2
    
    # Case 4: both recency and performance weighting
    performance_col = f"{target}_average_rolling_{rol_window}"
    points = data[performance_col]
    weights_1 = np.exp(alpha_recency * np.arange(n_samples) / n_samples)
    weights_1 /= np.sum(weights_1)
    weights_2 = np.where(
        points >= 1,
        np.exp(alpha_points * points),
        np.maximum(1 + alpha_points * points, 0.1)
    )
    weights_2 /= np.sum(weights_2)
    
    combined_weights = alpha_combined * weights_1 + (1 - alpha_combined) * weights_2
    combined_weights /= np.sum(combined_weights)

    return combined_weights

def calculate_adjusted_predictions(
        predictions: dict[str, np.ndarray], 
        datasets: dict[str, pd.DataFrame],
        form_window: int = 4,
) -> dict[str, dict]:
    """
    Calculate various adjusted predictions based on minutes and form columns if they exist
    """
    adjusted_results = {}

    adjustment_types = [
            ("minutes_adjusted", f"minutes_average_rolling_{form_window}_normalized"),
            ("form_adjusted", f"form_average_rolling_{form_window}_normalized"),
            ("form_sqrt_adjusted", f"form_average_rolling_{form_window}_sqrt_normalized")
        ]

    for adj_name, col_name in adjustment_types:
        adjusted_results[adj_name] = {}
        for dataset_name, pred in predictions.items():
            dataset = datasets[dataset_name]

            if col_name in dataset.columns:
                adjusted_results[adj_name][dataset_name] = pred * dataset[col_name]
            else:
                adjusted_results[adj_name][dataset_name] = pred
    
    return adjusted_results
    