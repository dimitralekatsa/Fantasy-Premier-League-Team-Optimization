"""
Feature definition for FPL prediction models
Organized by window size and feature types
"""

# Base features (always included without average calculations) 
base_features = [
    "was_home",
    "penalties_order",
    "corners_and_indirect_freekicks_order",
    "direct_freekicks_order",
    "team_strength",
    "team_strength_overall",
    "team_strength_attack",
    "team_strength_defence",
    "opponent_difficulty",
    "team_difficulty",
    "is_late_season",
    "team_total_points",
    "importance_rank",
    "importance",
    "cumulative_points",
]

stats = [
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "creativity",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals",
    "expected_goals_conceded",
    "goals_scored",
    "goals_conceded",
    "ict_index",
    "influence",
    "minutes",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "saves",
    "threat",
    "total_points",
    "yellow_cards",
]

def get_rolling_features(window: int) -> list:
    """
    Generate rolling averages feature names
    """
    return [f"{stat}_average_rolling_{window}" for stat in stats]

def get_difficulty_features(window: int) -> list:
    """
    Generate difficulty-based rolling averages names
    """
    return [f"{stat}_average_difficulty_{window}" for stat in stats]

def get_team_features(window: int) -> list:
    """
    Generate team-based rolling averages names
    """
    return [
        f"team_goals_scored_rolling_{window}",
        f"team_goals_conceded_rolling_{window}",
        f"opponent_goals_scored_rolling_{window}",
        f"opponent_goals_conceded_rolling_{window}",
        f"position_team_stat_rolling_{window}",
        f"position_opponent_stat_rolling_{window}",
    ]

def get_form_features(window: int) -> list:
    """
    Generate form rolling averages names
    """
    return [
        f"form_average_rolling_{window}",
        f"minutes_average_rolling_{window}",
        f"form_average_rolling_{window}_normalized",
        f"form_average_rolling_{window}_sqrt_normalized",
        f"minutes_average_rolling_{window}_normalized",
    ]

def get_scalable_features(rolling_window: int, difficulty_window: int) -> list:
    """
    Get features that should be scaled based on window parameters
    """
    scalable_features = []

    scalable_features.extend(get_rolling_features(window=rolling_window))
    scalable_features.extend(get_difficulty_features(window=difficulty_window))

    return scalable_features

def get_feature_set(rolling_window: int, difficulty_window: int, team_window: int, form_window: int, scaled_features: bool = True) -> list:
    """
    Get complete list of features for training
    """
    features = []

    features.extend(base_features)
    
    features.extend(get_team_features(window=team_window))
    features.extend(get_form_features(window=form_window))

    if scaled_features:
        scalable_features = get_scalable_features(rolling_window=rolling_window, difficulty_window=difficulty_window)
        features.extend([f"{feature}_scaled" for feature in scalable_features])
    else:
        features.extend(get_rolling_features(window=rolling_window))
        features.extend(get_difficulty_features(window=difficulty_window))

    return features

# Predefined combinations (_r{rolling}_d{difficulty_t{team}_f{form}})
features_r5_d4_t8_f5 = get_feature_set(
    rolling_window=5,
    difficulty_window=4,
    team_window=8,
    form_window=5,
    scaled_features=True
)

features_r8_d4_t4_f3 = get_feature_set(
    rolling_window=8,
    difficulty_window=4,
    team_window=4,
    form_window=3,
    scaled_features=True
)

feature_groups = {
    "random_forest": features_r5_d4_t8_f5,
    "gbm": features_r8_d4_t4_f3,
    "lgbm": features_r8_d4_t4_f3,
}
