"""
FPL Prediction Pipeline - Main Script
Orchestrates data loading, feature engineering, model training, prediction and team optimization
"""

import os
import yaml
import logging
import pandas as pd

import fpl.features.features as features
from fpl.data.fetch_past_seasons_data import (
    FPLPastSeasonProcessor,
)
from fpl.data.fetch_current_data import (
    FPLPastFixturesFetcher,
    FPLPastFixturesProcessor,
    FPLFutureFixturesFetcher,
    FPLFutureFixturesProcessor,
)
from fpl.features.feature_engineering import (
    FPLPastFeatureEngineering,
    FPLFutureFeatureEngineering,
)
from fpl.models.rf_models import (
    RandomForest,
)
from fpl.models.gbm_models import (
    GradientBoosting,
)
from fpl.models.lgbm_models import (
    LightGradientBoosting,
)
from fpl.utils.utils import (
    get_current_gameweek,
    handle_double_gameweeks,
    transform_to_wide_format,
)
from fpl.optimization.team_selection import (
    WildcardOptimization,
    FreeHitOptimization,
)
from fpl.data.fetch_managers_teams import (
    FPLTeamFetcher,
)
from fpl.optimization.transfer_engines import (
    BeamSearchFPLOptimizer,
    GreedyFPLTransferOptimizer,
)

class ConfigLoader:
    """
    Load configuration from a YAML file.
    """
    @staticmethod
    def load_config(config_path: str = "config.yaml") -> dict:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Create directories if they do not exist
        for path_key in ["data_dir", "models_dir", "results_dir", "predictions_dir"]:
            path = config["paths"][path_key]
            os.makedirs(path, exist_ok=True)

        # Process feature configurations
        config = ConfigLoader._process_feature_config(config)

        return config
    
    @staticmethod
    def _process_feature_config(config: dict) -> dict:
        """
        Process feature configurations to resolve refernces to features.py
        """
        rolling_window = config["feature_engineering"]["rolling_window"]
        difficulty_window = config["feature_engineering"]["difficulty_window"]
        team_window = config["feature_engineering"]["team_window"]
        form_window = config["feature_engineering"]["form_window"]

        # Determine if we should scale features
        scale_features = config["feature_engineering"].get("scale_features", True)

        if config["feature_engineering"].get("feature_set") != "auto" or config["feature_engineering"].get("columns_to_scale") not in [None, "auto"]:
            # Validate feature references and warn about mismatched windows
            ConfigLoader._validate_feature_references(
                config["feature_engineering"].get("feature_set"), "feature_set",
                rolling_window, difficulty_window, team_window, form_window
            )
            ConfigLoader._validate_feature_references(
                config["feature_engineering"].get("columns_to_scale"), "columns_to_scale",
                rolling_window, difficulty_window, team_window, form_window
            )
        
        config["feature_engineering"]["feature_set"] = features.get_feature_set(
            rolling_window=rolling_window,
            difficulty_window=difficulty_window,
            team_window=team_window,
            form_window=form_window,
            scaled_features=scale_features
        )

        if scale_features:
            config["feature_engineering"]["columns_to_scale"] = features.get_scalable_features(
                rolling_window=rolling_window,
                difficulty_window=difficulty_window
            )
        else:
            config["feature_engineering"]["columns_to_scale"] = None
        
        # Process Optuna storage path
        optuna_file = config["paths"]["optuna_storage"]
        if not optuna_file.startswith("sqlite:///"):
            db_path = os.path.abspath(optuna_file)
            config["paths"]["optuna_storage"] = f"sqlite:///{db_path}"
        
        return config
    
    @staticmethod
    def _validate_feature_references(feature_ref, list_name: str, rolling_window: int,
                                     difficulty_window: int, team_window: int, form_window: int):
        """
        Validate feature references.
        Warn if any window parameters do not match the expected values for certain feature types
        """
        if not isinstance(feature_ref, str):
            return
        
        # Check parameterized feature names
        if ConfigLoader._is_parameterized_feature(feature_ref):
            parsed_params = ConfigLoader._parse_parameterized_feature(feature_ref)

            mismatched_params = []
            if parsed_params.get("r") != rolling_window:
                mismatched_params.append(f"rolling_window (config: {rolling_window}, got: {parsed_params.get('r')})")
            if parsed_params.get("d") != difficulty_window:
                mismatched_params.append(f"difficulty_window (config: {difficulty_window}, got: {parsed_params.get('d')})")
            if parsed_params.get("t") != team_window and parsed_params.get("t") is not None:
                mismatched_params.append(f"team_window (config: {team_window}, got: {parsed_params.get('t')})")
            if parsed_params.get("f") != form_window and parsed_params.get("f") is not None:
                mismatched_params.append(f"form_window (config: {form_window}, got: {parsed_params.get('f')})")
            
            if mismatched_params:
                logging.warning(f"Feature set '{list_name}' has mismatched parameters: {', '.join(mismatched_params)}. \nPlease ensure the parameters match the configured window values.")
                logging.warning(f"   Name suggests: r={parsed_params.get('r')}, d={parsed_params.get('d')}, "
                      f"t={parsed_params.get('t')}, f={parsed_params.get('f')}")
                logging.warning(f"   Config uses:   r={rolling_window}, d={difficulty_window}, "
                      f"t={team_window}, f={form_window}")
                logging.warning(f"   -> Using CONFIG parameters (r={rolling_window}, d={difficulty_window}, t={team_window}, f={form_window})")
                logging.warning(f"   -> To fix: Either use 'auto' or update config to match the name\n")
        
        elif hasattr(features, "feature_groups") and feature_ref in features.feature_groups:
            actual_feature_name = ConfigLoader._get_feature_set_name(feature_ref)

            if actual_feature_name and ConfigLoader._is_parameterized_feature(actual_feature_name):
                parsed_params = ConfigLoader._parse_parameterized_feature(actual_feature_name)
                mismatched_params = []

                if parsed_params.get("r") != rolling_window:
                    mismatched_params.append(f"rolling_window (config: {rolling_window}, got: {parsed_params.get('r')})")
                if parsed_params.get("d") != difficulty_window:
                    mismatched_params.append(f"difficulty_window (config: {difficulty_window}, got: {parsed_params.get('d')})")
                if parsed_params.get("t") != team_window and parsed_params.get("t") is not None:
                    mismatched_params.append(f"team_window (config: {team_window}, got: {parsed_params.get('t')})")
                if parsed_params.get("f") != form_window and parsed_params.get("f") is not None:
                    mismatched_params.append(f"form_window (config: {form_window}, got: {parsed_params.get('f')})")
            
            
                if mismatched_params:
                    logging.warning(f"Feature set '{list_name}' has mismatched parameters: {', '.join(mismatched_params)}. \nPlease ensure the parameters match the configured window values.")
                    logging.warning(f"   {feature_ref} uses: r={parsed_params.get('r')}, d={parsed_params.get('d')}, "
                        f"t={parsed_params.get('t')}, f={parsed_params.get('f')}")
                    logging.warning(f"   Config uses:           r={rolling_window}, d={difficulty_window}, "
                        f"t={team_window}, f={form_window}")
                    logging.warning(f"   -> Using CONFIG parameters (r={rolling_window}, d={difficulty_window}, t={team_window}, f={form_window})")
                    logging.warning(f"   -> To fix: Either use 'auto' or update config to match the detected windows\n")
        
        elif hasattr(features, feature_ref):
            # If it is a direct reference, validate it
            if ConfigLoader._is_parameterized_feature(feature_ref):
                parsed_params = ConfigLoader._parse_parameterized_feature(feature_ref)
                mismatched_params = []

                if parsed_params.get("r") != rolling_window:
                    mismatched_params.append(f"rolling_window (config: {rolling_window}, got: {parsed_params.get('r')})")
                if parsed_params.get("d") != difficulty_window:
                    mismatched_params.append(f"difficulty_window (config: {difficulty_window}, got: {parsed_params.get('d')})")
                if parsed_params.get("t") != team_window and parsed_params.get("t") is not None:
                    mismatched_params.append(f"team_window (config: {team_window}, got: {parsed_params.get('t')})")
                if parsed_params.get("f") != form_window and parsed_params.get("f") is not None:
                    mismatched_params.append(f"form_window (config: {form_window}, got: {parsed_params.get('f')})")
                
                if mismatched_params:
                    logging.warning(f"Feature set '{list_name}' has mismatched parameters: {', '.join(mismatched_params)}. \nPlease ensure the parameters match the configured window values.")
                    logging.warning(f"   Name suggests: r={parsed_params.get('r')}, d={parsed_params.get('d')}, "
                          f"t={parsed_params.get('t')}, f={parsed_params.get('f')}")
                    logging.warning(f"   Config uses:   r={rolling_window}, d={difficulty_window}, "
                          f"t={team_window}, f={form_window}")
                    logging.warning(f"   -> Using CONFIG parameters (r={rolling_window}, d={difficulty_window}, t={team_window}, f={form_window})")
                    logging.warning(f"   -> To fix: Either use 'auto' or update config to match the name\n")

    @staticmethod
    def _get_feature_set_name(feature_ref: str) -> str:
        """
        Get the actual feature set name from features.feature_groups
        """
        if not hasattr(features, "feature_groups"):
            return None
        
        # Check if this group maps to a variable in features.py
        features_list = features.feature_groups.get(feature_ref)
        if features_list is None:
            return None
        
        for attr_name in dir(features):
            if not attr_name.startswith("_"):  # Skip private attributes
                try:
                    attr_value = getattr(features, attr_name)
                    if attr_value is features_list and ConfigLoader._is_parameterized_feature(attr_name):
                        return attr_name
                except:
                    continue
        
        return None
    
    @staticmethod
    def _is_parameterized_feature(feature_name: str) -> bool:
        """
        Check if a feature name follows the parameterized pattern
        e.g. features_r5_d4_t8_f5
        """
        return (feature_name.startswith("features_r") and "_d" in feature_name) or \
               (feature_name.startswith("columns_r") and "_d" in feature_name)
    
    @staticmethod
    def _parse_parameterized_feature(feature_name: str) -> dict:
        """
        Parse a parameterized feature name into its components
        e.g. features_r5_d4_t8_f5 -> {'r': 5, 'd': 4, 't': 8, 'f': 5}
        """
        params = {}
        try:
            parts = feature_name.split("_")
            for part in parts:
                if part.startswith("r") and len(part) > 1:
                    params["r"] = int(part[1:])
                elif part.startswith("d") and len(part) > 1:
                    params["d"] = int(part[1:])
                elif part.startswith("t") and len(part) > 1:
                    params["t"] = int(part[1:])
                elif part.startswith("f") and len(part) > 1:
                    params["f"] = int(part[1:])
        except Exception as e:
            logging.warning(f"Failed to parse parameterized feature name '{feature_name}': {e}")
        
        return params

    
class FPLPipeline:
    """
    Main FPL prediction pipeline that orchestrates the entire workflow
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigLoader.load_config(config_path)
        self._setup_logging()

        self.past_processor = None
        self.past_seasons_data = None

        self.current_fetcher = None
        self.current_processor = None
        self.past_data = None
        self.feature_engineer = None
        self.training_data = None

        self.future_fetcher = None
        self.future_processor = None
        self.future_data = None
        self.future_feature_engineer = None

        self.model = None
        self.predictions = None

        self.logger.info("FPL Pipeline initialized")

    def _setup_logging(self):
        """
        Setup logging configuration
        """
        log_config = self.config["logging"]
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[
                logging.FileHandler(log_config.get("file", "fpl_pipeline.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("FPLPipeline")

    def fetch_past_data(self, seasons: list | None = None) -> pd.DataFrame:
        """
        Fetch historical FPL data for specified seasons.
        If you use season '20-21', you have to remove the columns that startwith 'expected_' from the feature set
        
        Parameters:
            seasons: List of seasons to fetch (e.g. ['20-21', '21-22', '22-23'])
        """
        self.logger.info("Fetching past seasons data...")

        if seasons is None:
            seasons = self.config["data"]["past_seasons"]
        
        self.logger.info(f"Using seasons: {seasons}")

        self.past_processor = FPLPastSeasonProcessor(
            path=self.config["paths"]["data_dir"],
            save_data=self.config["data"].get("save_data", True)
        )

        processed_seasons_dict, self.past_seasons_data = self.past_processor.process_multiple_seasons(seasons=seasons)

        return self.past_seasons_data
        
    def fetch_current_data(self) -> pd.DataFrame:
        """
        Fetch current season data
        """
        self.logger.info("Fetching current season data...")

        self.current_fetcher = FPLPastFixturesFetcher(
            path=self.config["paths"]["data_dir"],
            save_data=self.config["data"].get("save_data", True),
            current_season=self.config["data"]["current_season"],
            hard_fetch=self.config["data"].get("hard_fetch", False),
            player_status=self.config["data"].get("player_status", ["a", "d", "i", "u"]),
        )
        self.current_processor = FPLPastFixturesProcessor(
            path=self.config["paths"]["data_dir"],
            save_data=self.config["data"].get("save_data", True),
            current_season=self.config["data"]["current_season"]            
        )

        existing_data, new_data = self.current_fetcher.fetch_incremental_player_history()
        new_df, current_season_df = self.current_processor.process_current_season_data(existing_data, new_data)

        try:
            past_data = pd.read_csv(os.path.join(self.config["paths"]["data_dir"], "training_dataset.csv"))
        except FileNotFoundError:
            past_data = pd.DataFrame()

        if past_data.empty:
            past_data = pd.concat([current_season_df, self.past_seasons_data], ignore_index=True)

        if not new_df.empty:
            new_gws = set(new_df.gameweek.unique())
            past_data = past_data[~((past_data.season == past_data.season.max()) & (past_data.gameweek.isin(new_gws)))]
            past_data = pd.concat([past_data, new_df], ignore_index=True)

        self.past_data = past_data[past_data.position != "Manager"]

        return self.past_data
    
    def process_past_data(self) -> pd.DataFrame:
        """
        Process historical data to create features for model training

        Return:
            self.training_data: Merged dataset with all selected past seasons and features for training
        """
        self.logger.info("Processing past data for feature engineering...")

        if self.past_data is None:
            raise ValueError("Past data not loaded. Please run fetch_past_data() and fetch_current_data() first.")

        self.feature_engineer = FPLPastFeatureEngineering(
            path=self.config["paths"]["data_dir"],
            save_data=self.config["data"].get("save_data", True)
        )

        # Determine columns to scale based on scale_features setting
        columns_to_scale = self.config["feature_engineering"]["columns_to_scale"] if self.config["feature_engineering"]["scale_features"] else None
        
        self.training_data = self.feature_engineer.run_pipeline(
            df=self.past_data,
            diff_window=self.config["feature_engineering"]["difficulty_window"],
            window=self.config["feature_engineering"]["rolling_window"],
            team_window=self.config["feature_engineering"]["team_window"],
            form_window=self.config["feature_engineering"]["form_window"],
            columns_to_scale=columns_to_scale,
            save_filename="training_dataset.csv"
        )

        return self.training_data
    
    def fetch_future_data(self) -> pd.DataFrame:
        """
        Fetch future fixtures data
        """
        self.logger.info("Fetching future fixtures data...")

        self.future_fetcher = FPLFutureFixturesFetcher(
            path=self.config["paths"]["data_dir"],
            current_season=self.config["data"]["current_season"],
            player_status=self.config["data"].get("player_status", ["a", "d", "i", "u"]),
            save_data=self.config["data"].get("save_data", True),
            horizon=5
        )
        self.future_processor = FPLFutureFixturesProcessor(
            path=self.config["paths"]["data_dir"],
            save_data=self.config["data"].get("save_data", True),
            current_season=self.config["data"]["current_season"]
        )

        future_fixtures = self.future_fetcher.fetch_future_fixtures_data()
        self.future_data = self.future_processor.process_future_fixtures_data(future_fixtures)

        return self.future_data
    
    def process_future_data(self) -> pd.DataFrame:
        """
        Process future fixtures data to create features for prediction

        Returns:
            self.future_data: Upcoming gameweeks dataframe
        """
        self.logger.info("Processing future fixtures data for feature engineering...")

        if self.future_data is None:
            raise ValueError("Future data not loaded. Please run fetch_future_data() first.")

        if self.training_data is None:
            raise ValueError("Training data not processed. Please run process_past_data() first.")

        self.future_feature_engineer = FPLFutureFeatureEngineering(
            path=self.config["paths"]["data_dir"],
            save_data=self.config["data"].get("save_data", True)
        )

        # Determine columns to scale based on scale_features setting
        columns_to_scale = self.config["feature_engineering"]["columns_to_scale"] if self.config["feature_engineering"]["scale_features"] else None

        self.future_data = self.future_feature_engineer.run_pipeline(
            past_df=self.training_data,
            future_df=self.future_data,
            diff_window=self.config["feature_engineering"]["difficulty_window"],
            window=self.config["feature_engineering"]["rolling_window"],
            team_window=self.config["feature_engineering"]["team_window"],
            form_window=self.config["feature_engineering"]["form_window"],
            columns_to_scale=columns_to_scale,
            save_filename="prediction_dataset.csv"
        )

        return self.future_data

    def train_model(self, feature_list: list | None = None, rolling_cols: list | None = None, difficulty_cols: list | None = None):
        """
        Train the specified model using the processed training data

        Parameters:
            feature_list: Optional custom feature list (uses config if None)
            rolling_cols: Optional custom rolling columns list (auto-generated if None)
            difficulty_cols: Optional custom difficulty rolling columns list (auto-generated if None)
        """
        self.logger.info("Training model...")

        if self.training_data is None:
            raise ValueError("Training data not processed. Please run process_past_data() first.")

        model_type = self.config["model"].get("model_type", "random_forest").lower()
        models_dir = self.config["paths"]["models_dir"]
        current_gw = get_current_gameweek()

        model_config = self.config["model"]
        sample_weights_config = model_config["sample_weights"]
        fe_config = self.config["feature_engineering"]
        common_params = {
            "categorical_features": model_config["categorical_features"],
            "alpha_recency": sample_weights_config["alpha_recency"],
            "alpha_points": sample_weights_config["alpha_points"],
            "alpha_combined": sample_weights_config["alpha_combined"],
            "form_window": fe_config["form_window"],
            "rol_window": fe_config["rolling_window"],
            "random_state": 42,
            "target_column": model_config["target_column"],
        }

        if rolling_cols is None:
            rolling_cols = features.get_rolling_features(window=fe_config["rolling_window"])
        
        if difficulty_cols is None:
            difficulty_cols = features.get_difficulty_features(window=fe_config["difficulty_window"])
        
        include_test_set = model_config["include_test_set"]
        save_results = True if include_test_set else False

        model_params = {
            "data": self.training_data,
            "features": fe_config["feature_set"] if feature_list is None else feature_list,
            "rolling_cols": rolling_cols,
            "difficulty_cols": difficulty_cols,
            "include_test_set": include_test_set,
            "save_results": save_results,
            "trials": model_config["hyperparameter_tuning"]["n_trials"],
            "csv_path": "models_results.csv",
        }

        if model_type in ["random_forest", "rf", "random forest"]:
            model = RandomForest(**common_params)
            model.train(study_name=f"rf_study(r{fe_config['rolling_window']}_d{fe_config['difficulty_window']}_t{fe_config['team_window']}_f{fe_config['form_window']})",
                        model_name=f"rf_(r{fe_config['rolling_window']}_d{fe_config['difficulty_window']}_t{fe_config['team_window']}_f{fe_config['form_window']})",
                        **model_params)
            model_filename = f"rf_model_gw{current_gw}.pkl"

        elif model_type in ["gradient_boosting", "gbm", "gb"]:
            model = GradientBoosting(**common_params)
            model.train(study_name=f"gbm_study(r{fe_config['rolling_window']}_d{fe_config['difficulty_window']}_t{fe_config['team_window']}_f{fe_config['form_window']})",
                            model_name=f"gbm_(r{fe_config['rolling_window']}_d{fe_config['difficulty_window']}_t{fe_config['team_window']}_f{fe_config['form_window']})",
                            **model_params)
            model_filename = f"gbm_model_gw{current_gw}.pkl"

        elif model_type in ["lightgbm", "lgbm", "lgb"]:
            model = LightGradientBoosting(**common_params)
            model.train(study_name=f"lgbm_study(r{fe_config['rolling_window']}_d{fe_config['difficulty_window']}_t{fe_config['team_window']}_f{fe_config['form_window']})",
                            model_name=f"lgbm_(r{fe_config['rolling_window']}_d{fe_config['difficulty_window']}_t{fe_config['team_window']}_f{fe_config['form_window']})",
                            **model_params)
            model_filename = f"lgbm_model_gw{current_gw}.pkl"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if model_config["save_model"]:
                model.save_model(os.path.join(models_dir, model_filename))

        self.model = model

        return self.model
    
    def generate_predictions(self) -> pd.DataFrame:
        """
        Generate predictions using the trained model

        Returns:
            self.predictions: Dataset for upcoming fixtures with players' "total_points" predictions
        """
        self.logger.info("Generating predictions...")

        predictions_dir = self.config["paths"]["predictions_dir"]
        upcoming_gw = get_current_gameweek() + 1

        predictions = self.model.predict(X=self.future_data, prediction_column="predicted_points")

        preds_filename = f"{self.config['model']['model_type']}_gw{upcoming_gw}.csv"
        predictions.to_csv(os.path.join(predictions_dir, preds_filename))

        self.predictions = handle_double_gameweeks(predictions)

        return self.predictions
    
    def optimize_transfers(self, team_id: int, budget: float | None = None):
        """
        Optimize transfers using specified engine
        """

        if self.predictions is None:
            raise ValueError("Predictions not generated, run generate_predictions() first.")
        
        engine = self.config["transfers"]["transfer_engine"]
        
        self.logger.info(f"Optimizing transfers using {engine} search engine...")

        team_fetcher = FPLTeamFetcher(team_id=team_id)
        team_data = team_fetcher.get_complete_team_data()
        print(f"Available free transfers: {self.config["initial_transfers"]}")

        wide_preds = transform_to_wide_format(self.predictions)

        transfers_config = self.config["transfers"]
        if budget is None:
            budget = transfers_config["budget"]

        initial = self.config["initial_transfers"]
        max_transfers = self.config["fpl_rules"]["max_transfers"]

        if isinstance(initial, int) and 1 <= initial <= max_transfers:
            available_transfers = initial
        else:
            available_transfers = team_data["available_free_transfers"]
        
        common_params = {
            "players_df": wide_preds,
            "current_team": team_data["player_ids"],
            "initial_transfers": available_transfers,
            "budget": budget,
            "valid_status": transfers_config["valid_status"],
            "debug": transfers_config["debug"],
            "points_prefix": "Points_GW",
            "team_size": self.config["fpl_rules"]["team_size"],
            "max_per_team": self.config["fpl_rules"]["max_per_team"],
            "max_transfers": max_transfers,
        }

        if engine == "beam":
            beam_config = transfers_config["beam"]
            beam = BeamSearchFPLOptimizer(beam_width=beam_config["beam_width"], **common_params)
            return beam.optimize_transfers()
        elif engine == "greedy":
            greedy_config = transfers_config["greedy"]
            greedy = GreedyFPLTransferOptimizer(max_threshold=greedy_config["max_threshold"], min_threshold=greedy_config["min_threshold"],
                                                threshold_step=greedy_config["threshold_step"], top_transfers_to_track=greedy_config["top_transfers_to_track"],
                                                must_have_threshold=greedy_config["must_have_threshold"], future_min_threshold=greedy_config["future_min_threshold"],
                                                **common_params)
            return greedy.optimize_transfers()
        else:
            raise ValueError(f"Unknown transfer engine: {engine}")
        
    def prediction_pipeline(self):
        """
        Run the complete pipeline from data fetching to predictions
        """
        self.fetch_past_data(seasons=None)

        self.fetch_current_data()
        past_gws_df = self.process_past_data()

        self.fetch_future_data()
        future_gws_df = self.process_future_data()

        # Drop rows that include players that are not part of the current season, so if the code is not in the current season drop the player
        if "code" in self.config["model"]["categorical_features"]:
            self.training_data = self.training_data[self.training_data.code.isin(self.training_data[self.training_data.season == self.training_data.season.max()].code.unique())]

        model = self.train_model()

        preds_df = self.generate_predictions()

        return past_gws_df, future_gws_df, model, preds_df

    def optimize_team(self, team_id: int | None, budget: float | None = None):
        """
        Optimize manager's team.
        For first gameweek the wildcard optimization is called, otherwise a transfer engine.
        You can use the wildcard or freehit optimization if you change the use_wildcard: true or use_freehit: true in config

        Returns:
            (dict): dictionary with different keys based on the optimization engine used
                type (str): "wildcard", "freehit", "transfer_engine"
                type-scecific keys:
                    wildcard:
                        - optimal_team (pd.DataFrame): full squad
                        - starters_upcoming_gw (pd.DataFrame): 11 starters for next gameweek
                        - updated_predictions (pd.DataFrame): predictions with in_team column
                    freehit:
                        - optimal_team (pd.DataFrame): optimal team for the gameweek
                    transfer_engine:
                        - team (list): list of player IDs in final team
                        - team_names (list): list of playes names in final team
                        - total_points (float): sum of total predicted points of final team
                        - transfer_plan (list): list of dicts, one per gameweek with transfer details
        """
        current_gw = get_current_gameweek()
        team_selection_config = self.config["team_selection"]
        
        common_params = {
            "predictions_df": self.predictions,
            "valid_status": team_selection_config["valid_status"],
            "points_col": "predicted_points",
            "max_per_team": self.config["fpl_rules"]["max_per_team"],
        }

        if current_gw == 1 or team_selection_config["use_wildcard"] or team_id is None:
            wildcard = WildcardOptimization(budget=team_selection_config["wildcard"]["budget"],
                                            starters_budget_pct=team_selection_config["wildcard"]["starters_budget_pct"],
                                            num_starters=team_selection_config["wildcard"]["num_starters"],
                                            future_weights=team_selection_config["wildcard"]["future_weights"],
                                            **common_params)
            optimal_team, starters_upcoming_gw, updated_predictions = wildcard.optimize_wildcard()
            return {
                "type": "wildcard",
                "optimal_team": optimal_team,
                "starters_upcoming_gw": starters_upcoming_gw,
                "updated_predictions": updated_predictions,
            }
        elif team_selection_config["use_freehit"]:
            freehit = FreeHitOptimization(budget=team_selection_config["freehit"]["budget"],
                                          starters_budget_pct=team_selection_config["freehit"]["starters_budget_pct"],
                                          **common_params)
            optimal_team = freehit.optimize_freehit()
            return {
                "type": "freehit",
                "optimal_team": optimal_team,
            }
        else:
            transfer_result = self.optimize_transfers(team_id=team_id, budget=budget)
            return {
                "type": "transfer_engine",
                **transfer_result     # Unpacks team, team_names, total_points, transfer_plan
            }
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FPL Prediction Pipeline")
    parser.add_argument("--team-id", type=int, help="Your FPL team ID")
    parser.add_argument("--budget", type=float, default=None, help="Budget for transfers")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip team optimization")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip pipeline, only run optimization")
    args = parser.parse_args()

    pipeline = FPLPipeline()

    if args.skip_pipeline:
        # Load existing predictions
        predictions_dir = pipeline.config["paths"]["predictions_dir"]
        upcoming_gw = get_current_gameweek() + 1
        preds_filename = f"{pipeline.config['model']['model_type']}_gw{upcoming_gw}.csv"
        pipeline.predictions = pd.read_csv(os.path.join(predictions_dir, preds_filename))
        pipeline.predictions = handle_double_gameweeks(pipeline.predictions)
    else:
        pipeline.prediction_pipeline()

    # Optionally optimize team
    if not args.skip_optimization:
        team_id = args.team_id if args.team_id is not None else pipeline.config.get("TEAM_ID")
        result = pipeline.optimize_team(team_id=team_id, budget=args.budget)
        print(f"Optimization Type: {result['type']}")
