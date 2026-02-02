import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import threading
import optuna
from optuna_dashboard import run_server
import os
import joblib
import matplotlib.pyplot as plt

# Import shared utilities
from fpl.utils.model_utils import (
    prepare_model_data, 
    transform_features,
    calculate_metrics,
    print_results,
    save_results_to_csv,
    calculate_sample_weights,
    calculate_adjusted_predictions,
)


class RandomForest:
    """
    Random Forest Trainer designed for FPL predictions implementing:
        - Sample weighting (recency + performance)
        - Hyperparameter optimization with Optuna
        - Minutes and form adjusted predictions
    
    alpha_recency >> more weight to recent samples
    alpha_points >> more weight to high scoring samples
    alpha_combined >> more weight to alpha_recency than alpha_points (0.5 = equal)
    
    Parameters:
        categorical_features: list of categorical feature names
        alpha_recency: recency weight factor for sample weighting
        alpha_points: performance weight factor for sample weighting
        alpha_combined: combined weight factor for sample weighting
        form_window: window size for form adjustment
        rol_window: window size for recency weighting
        random_state: random seed for reproducibility
        target_column: name of the target variable column
    """
    def __init__(
            self,
            categorical_features: list[str] = ["team_code", "opponent_team_code", "position"],
            alpha_recency: float = 0,
            alpha_points: float = 0,
            alpha_combined: float = 0,
            form_window: int = 4,
            rol_window: int = 8,
            random_state: int = 42,
            target_column: str = "total_points",
    ):
        
        self.categorical_features = categorical_features
        self.alpha_recency = alpha_recency
        self.alpha_points = alpha_points
        self.alpha_combined = alpha_combined
        self.form_window = form_window
        self.rol_window = rol_window
        self.random_state = random_state
        self.target = target_column

        self.model = None
        self.study = None
        self.best_params = None
        self.feature_names = None
        self.active_categorical_features = None
        self.transformer = None
        
    def _setup_optuna_study(self, study_name: str = "RandomForest") -> optuna.Study:
        """
        Set up Optuna study for hyperparameter optimization with dashboard support
        """
        storage_path = "sqlite:///models.db"

        # Start dashboard in background
        def start_dashboard():
            try:
                run_server(storage=storage_path, port=8080)
            except:
                pass

        threading.Thread(target=start_dashboard, daemon=True).start()
        
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage_path,
            load_if_exists=True,
        )

        return study
    
    def _objective_function(
            self,
            trial: optuna.Trial,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
            sample_weights: np.ndarray | None,
    ) -> float:
        """
        Optuna objective function for hyperparameter optimization
        """
        use_max_leaf_nodes = trial.suggest_categorical("use_max_leaf_nodes", [True, False])
        
        params = {
            # Core parameters
            "n_estimators": trial.suggest_int("n_estimators", 100, 900),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse"]),
            # Regularization parameters
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.1),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.4, 0.6, 0.8, None]),
            "min_samples_split": trial.suggest_float("min_samples_split", 0.003, 0.06),
            "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.005, 0.05),
            # Bootstrap and sampling
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            # Control overfitting
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.1),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.1),
            # Fixed parameters
            "random_state": self.random_state,
            "n_jobs": -1,
        }

        # Conditionally add max_leaf_nodes (but don't include the helper variable)
        if use_max_leaf_nodes:
            params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 10, 1000)

        # Conditionally add max_samples
        if params["bootstrap"]:
            params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        val_preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, val_preds)

        return rmse
    
    def _calculate_metrics(self, df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, metric_prefix: str = "") -> dict:
        """
        Calculate MAE and RMSE for all players, top players and by position
        """
        metrics = {}

        # Calculate metrics for all players
        metrics.update(calculate_metrics(y_true=y_true, y_pred=y_pred, prefix=metric_prefix))

        # Top players metrics (total points >=4)
        high_value_mask = y_true >= 4
        if high_value_mask.any():
            y_true_top = y_true[high_value_mask]
            y_pred_top = y_pred[high_value_mask]

            metrics.update(calculate_metrics(y_true=y_true_top, y_pred=y_pred_top, prefix=f"{metric_prefix}top_"))
        else:
            metrics[f"{metric_prefix}mae_top"] = np.nan
            metrics[f"{metric_prefix}rmse_top"] = np.nan

        for pos in df.position.unique():
            pos_mask = df.position == pos
            y_true_pos = y_true[pos_mask]
            y_pred_pos = y_pred[pos_mask]

            if len(y_true_pos) > 0:
                metrics.update(calculate_metrics(y_true=y_true_pos, y_pred=y_pred_pos, prefix=f"{metric_prefix}{pos}_"))
            else:
                metrics[f"{metric_prefix}mae_{pos}"] = np.nan
                metrics[f"{metric_prefix}rmse_{pos}"] = np.nan
        
        return metrics
        
    def train(
            self,
            data: pd.DataFrame,
            features: list[str],
            rolling_cols: list[str],
            difficulty_cols: list[str],
            include_test_set: bool = False,
            trials: int = 100,
            study_name: str | None = None,
            model_name: str | None = None,
            save_results: bool = True,
            csv_path: str = "models_results.csv",
    ) -> dict:
        """
        Parameters:
            trials: number of optuna optimization trials
            study_name: name of the optuna study
            model_name: name of the model for csv saving
        """
        if include_test_set:
            (X_train, y_train), (X_val, y_val), (X_test, y_test), transformer, updated_features, active_categorical, train_set, val_set, test_set = prepare_model_data(
                data=data, features=features, rolling_cols=rolling_cols, difficulty_cols=difficulty_cols,
                categorical_features=self.categorical_features, include_test_set=True,
                target_column=self.target, random_state=self.random_state
            )
        else:
            (X_train, y_train), (X_val, y_val), transformer, updated_features, active_categorical, train_set, val_set = prepare_model_data(
                data=data, features=features, rolling_cols=rolling_cols, difficulty_cols=difficulty_cols,
                categorical_features=self.categorical_features, include_test_set=False,
                target_column=self.target, random_state=self.random_state
            )
            X_test, y_test, test_set = None, None, None
        
        # Store components
        self.transformer = transformer
        self.feature_names = updated_features
        self.active_categorical_features = active_categorical

        sample_weights = calculate_sample_weights(
            data=train_set, alpha_recency=self.alpha_recency, alpha_points=self.alpha_points,
            alpha_combined=self.alpha_combined, rol_window=self.rol_window, target=self.target
        )

        study_name = study_name or f"rf_study_{len(features)}features"
        self.study = self._setup_optuna_study(study_name=study_name)

        objective = lambda trial: self._objective_function(
            trial=trial, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, sample_weights=sample_weights
        )

        self.study.optimize(objective, n_trials=trials, show_progress_bar=True, catch=(Exception,))
        self.best_params = {k: v for k, v in self.study.best_params.items() 
                   if k not in ["use_max_leaf_nodes"]}

        self.model = RandomForestRegressor(
            **self.best_params,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train, sample_weight=sample_weights)

        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        test_preds = self.model.predict(X_test) if X_test is not None else None

        predictions = {
            "train": train_preds,
            "val": val_preds,
        }
        
        datasets = {
            "train": train_set,
            "val": val_set,
        }
        
        targets = {
            "train": y_train,
            "val": y_val,
        }

        if include_test_set:
            predictions["test"] = test_preds
            datasets["test"] = test_set
            targets["test"] = y_test

        base_metrics = {}
        for dataset_name in targets.keys():
            metrics = self._calculate_metrics(
                df=datasets[dataset_name],
                y_true=targets[dataset_name],
                y_pred=predictions[dataset_name],
                metric_prefix=f"{dataset_name}_",
            )
            base_metrics.update(metrics)

        adjusted_predictions = calculate_adjusted_predictions(
            predictions=predictions, datasets=datasets, form_window=self.form_window
        )

        adj_metrics = {}
        for adj_type, adj_preds_dict in adjusted_predictions.items():
            adj_metrics[adj_type] = {}
            for dataset_name in targets.keys():
                metrics = self._calculate_metrics(
                    df=datasets[dataset_name],
                    y_true=targets[dataset_name],
                    y_pred=adj_preds_dict[dataset_name],
                    metric_prefix=f"{dataset_name}_",
                )
                adj_metrics[adj_type].update(metrics)

        results = {
            "model": self.model,
            "study": self.study,
            "best_params": self.best_params,
            "predictions": predictions,
            "adjusted_predictions": adjusted_predictions,
            "base_metrics": base_metrics,
            **adj_metrics,
        }

        print_results(results=results)

        if save_results:
            if model_name is None:
                model_name = f"rf_model_{len(features)}features"

            # Model params dictionary
            model_params = {
                "categorical_features": len(self.active_categorical_features),
                "features": len(self.feature_names),
                "alpha_recency": self.alpha_recency,
                "alpha_points": self.alpha_points,
                "alpha_combined": self.alpha_combined,
                "form_window": self.form_window,
                "rol_window": self.rol_window,
            }

            save_results_to_csv(results=results, model_name=model_name, model_params=model_params, csv_path=csv_path)
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """
        Save the complete model including transformer and metadata
        """
        if self.model is None:
            raise ValueError("No trained model to save. Please train the model first.")
        
        model_data = {
            "model": self.model,
            "transformer": self.transformer,
            "feature_names": self.feature_names,
            "active_categorical_features": self.active_categorical_features,
            "categorical_features": self.categorical_features,
            "alpha_recency": self.alpha_recency,
            "alpha_points": self.alpha_points,
            "alpha_combined": self.alpha_combined,
            "form_window": self.form_window,
            "rol_window": self.rol_window,
            "random_state": self.random_state,
            "study": self.study,
            "best_params": self.best_params
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.transformer = model_data["transformer"]
        self.feature_names = model_data["feature_names"]
        self.active_categorical_features = model_data["active_categorical_features"]
        self.categorical_features = model_data["categorical_features"]
        self.alpha_recency = model_data["alpha_recency"]
        self.alpha_points = model_data["alpha_points"]
        self.alpha_combined = model_data["alpha_combined"]
        self.form_window = model_data["form_window"]
        self.rol_window = model_data["rol_window"]
        self.random_state = model_data["random_state"]
        self.study = model_data.get("study")
        self.best_params = model_data.get("best_params")
        
        print(f"Model loaded from {filepath}")
    
    @classmethod
    def from_saved_model(cls, filepath: str) -> "RandomForest":
        """
        Create a new instance from a saved model
        """
        instance = cls()
        instance.load_model(filepath)
        return instance
    
    def predict(self, X: pd.DataFrame, prediction_column: str = "predicted_points") -> pd.DataFrame:
        """
        Predict using the trained Random Forest model

        Parameters:
            X: Input DataFrame
            prediction_column: Name of the column the predictions will be inserted
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call the train or load_model method first.")
        
        if self.transformer is None:
            raise ValueError("Transformer not found.")
        
        X_copy = X.copy()

        X_transformed, _, _, _ = transform_features(
            data=X_copy, features=self.feature_names, categorical_features=self.categorical_features,
            transformer=self.transformer
        )

        X_selected = X_transformed[self.feature_names]

        # Check for missing values
        missing_mask = X_selected.isnull().any(axis=1)
        
        if missing_mask.any():
            print(f"Warning: {missing_mask.sum()} rows have missing values in required features")
            print("These rows will have NaN predictions")

        predictions = np.full(len(X_selected), np.nan)  # Initialize with NaN
        valid_indices = ~missing_mask

        if valid_indices.any():
            preds = self.model.predict(X_selected[valid_indices])
            predictions[valid_indices] = preds
        
        X_copy[prediction_column] = predictions

        return X_copy
        
    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from the trained Random Forest model and plot top features
        
        Parameters:
            top_n: Number of top features to display in plot
            
        Returns:
            feature_inp: DataFrame with feature importance sorted by importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call the train method first.")
        
        # Random Forest only has feature_importances_ attribute (Gini importance)
        importance = self.model.feature_importances_
        
        feature_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values(by="importance", ascending=False)
        
        # Plot top_n features
        top_features = feature_imp.head(top_n)
        
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(top_features)), top_features["importance"], color="steelblue", alpha=0.7)
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Feature Importance")
        plt.title(f"Most Important Features of Random Forest Model")
        plt.gca().invert_yaxis()
        plt.gca().set_facecolor("none")
        plt.grid(axis="x", alpha=0.4)
        plt.tight_layout()
        plt.show()

        return feature_imp
    

class RandomForestAdjusted:
    """
    Random Forest Trainer designed for FPL predictions implementing:
        - Hyperparameter optimization with Optuna
        - Minutes or form adjusted training
    
    Parameters:
        categorical_features: list of categorical feature names
        adjusted_col: column name for minutes/form adjustment
        random_state: random seed for reproducibility
        target_column: name of the target variable column
    """
    def __init__(
            self,
            categorical_features: list[str] = ["team_code", "opponent_team_code", "position"],
            adjusted_col: str = "form_average_rolling_4_normalized",
            random_state: int = 42,
            target_column: str = "total_points",
    ):
        
        self.categorical_features = categorical_features
        self.adjusted_col = adjusted_col
        self.random_state = random_state
        self.target = target_column
        
        self.model = None
        self.study = None
        self.best_params = None
        self.feature_names = None
        self.active_categorical_features = None
        self.transformer = None

    def _setup_optuna_study(self, study_name: str = "RandomForestAdjusted") -> optuna.Study:
        """
        Set up Optuna study for hyperparameter optimization with dashboard support
        """
        storage_path = "sqlite:///models.db"

        # Start dashboard in background
        def start_dashboard():
            try:
                run_server(storage=storage_path, port=8080)
            except:
                pass

        threading.Thread(target=start_dashboard, daemon=True).start()
        
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=storage_path,
            load_if_exists=True,
        )

        return study
    
    def _objective_function(
            self,
            trial: optuna.Trial,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series
    ) -> float:
        """
        Optuna objective function for hyperparameter optimization
        """
        use_max_leaf_nodes = trial.suggest_categorical("use_max_leaf_nodes", [True, False])
        
        params = {
            # Core parameters
            "n_estimators": trial.suggest_int("n_estimators", 100, 900),
            "max_depth": trial.suggest_int("max_depth", 6, 30),
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "absolute_error", "friedman_mse"]),
            # Regularization parameters
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.1),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.4, 0.6, 0.8, None]),
            "min_samples_split": trial.suggest_float("min_samples_split", 0.003, 0.06),
            "min_samples_leaf": trial.suggest_float("min_samples_leaf", 0.005, 0.05),
            # Bootstrap and sampling
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            # Control overfitting
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.1),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.1),
            # Fixed parameters
            "random_state": self.random_state,
            "n_jobs": -1,
        }

        # Conditionally add max_leaf_nodes
        if use_max_leaf_nodes:
            params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 10, 1000)

        # Conditionally add max_samples
        if params["bootstrap"]:
            params["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, val_preds)

        return rmse
    
    def _calculate_metrics(self, df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, metric_prefix: str = "") -> dict:
        """
        Calculate MAE and RMSE for all players, top players and by position
        """
        metrics = {}

        # Calculate metrics for all players
        metrics.update(calculate_metrics(y_true=y_true, y_pred=y_pred, prefix=metric_prefix))

        # Top players metrics (total points >=4)
        high_value_mask = y_true >= 4
        if high_value_mask.any():
            y_true_top = y_true[high_value_mask]
            y_pred_top = y_pred[high_value_mask]

            metrics.update(calculate_metrics(y_true=y_true_top, y_pred=y_pred_top, prefix=f"{metric_prefix}top_"))
        else:
            metrics[f"{metric_prefix}mae_top"] = np.nan
            metrics[f"{metric_prefix}rmse_top"] = np.nan

        for pos in df.position.unique():
            pos_mask = df.position == pos
            y_true_pos = y_true[pos_mask]
            y_pred_pos = y_pred[pos_mask]

            if len(y_true_pos) > 0:
                metrics.update(calculate_metrics(y_true=y_true_pos, y_pred=y_pred_pos, prefix=f"{metric_prefix}{pos}_"))
            else:
                metrics[f"{metric_prefix}mae_{pos}"] = np.nan
                metrics[f"{metric_prefix}rmse_{pos}"] = np.nan
        
        return metrics
        
    def train(
            self,
            data: pd.DataFrame,
            features: list[str],
            rolling_cols: list[str],
            difficulty_cols: list[str],
            include_test_set: bool = False,
            trials: int = 100,
            study_name: str | None = None,
            model_name: str | None = None,
            save_results: bool = True,
            csv_path: str = "models_results.csv",
    ) -> dict:
        """
        Parameters:
            trials: number of optuna optimization trials
            study_name: name of the optuna study
            model_name: name of the model for csv saving
        """
        if include_test_set:
            (X_train, y_train), (X_val, y_val), (X_test, y_test), transformer, updated_features, active_categorical, train_set, val_set, test_set = prepare_model_data(
                data=data, features=features, rolling_cols=rolling_cols, difficulty_cols=difficulty_cols,
                categorical_features=self.categorical_features, include_test_set=True,
                target_column=self.target, random_state=self.random_state
            )

            if self.adjusted_col in train_set.columns:
                y_train = y_train * train_set[self.adjusted_col]
                y_val = y_val * val_set[self.adjusted_col]

        else:
            (X_train, y_train), (X_val, y_val), transformer, updated_features, active_categorical, train_set, val_set = prepare_model_data(
                data=data, features=features, rolling_cols=rolling_cols, difficulty_cols=difficulty_cols,
                categorical_features=self.categorical_features, include_test_set=False,
                target_column=self.target, random_state=self.random_state
            )
            X_test, y_test, test_set = None, None, None

            if self.adjusted_col in data.columns:
                y_train = y_train * train_set[self.adjusted_col]
        
        # Store components
        self.transformer = transformer
        self.feature_names = updated_features
        self.active_categorical_features = active_categorical

        parts = self.adjusted_col.split("_")
        base_type = parts[0]
        if "sqrt" in self.adjusted_col:
            adj_type = f"{base_type}_sqrt"
        else:
            adj_type = base_type
        study_name = study_name or f"rf_{adj_type}_study_{len(features)}features"
        self.study = self._setup_optuna_study(study_name=study_name)

        objective = lambda trial: self._objective_function(
            trial=trial, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
        )

        self.study.optimize(objective, n_trials=trials, show_progress_bar=True, catch=(Exception,))
        self.best_params = {k: v for k, v in self.study.best_params.items() 
                   if k not in ["use_max_leaf_nodes"]}

        self.model = RandomForestRegressor(
            **self.best_params,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        test_preds = self.model.predict(X_test) if X_test is not None else None

        predictions = {
            "train": train_preds,
            "val": val_preds,
        }
        
        datasets = {
            "train": train_set,
            "val": val_set,
        }
        
        targets = {
            "train": y_train,
            "val": y_val,
        }

        if include_test_set:
            predictions["test"] = test_preds
            datasets["test"] = test_set
            targets["test"] = y_test

        base_metrics = {}
        for dataset_name in targets.keys():
            metrics = self._calculate_metrics(
                df=datasets[dataset_name],
                y_true=targets[dataset_name],
                y_pred=predictions[dataset_name],
                metric_prefix=f"{dataset_name}_",
            )
            base_metrics.update(metrics)

        results = {
            "model": self.model,
            "study": self.study,
            "best_params": self.best_params,
            "predictions": predictions,
            "base_metrics": base_metrics,
        }

        print_results(results=results)

        if save_results:
            if model_name is None:
                model_name = f"rf_{adj_type}_model_{len(features)}features"

            # Model params dictionary
            model_params = {
                "categorical_features": len(self.active_categorical_features),
                "features": len(self.feature_names),
            }

            save_results_to_csv(results=results, model_name=model_name, model_params=model_params, csv_path=csv_path)
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """
        Save the complete model including transformer and metadata
        """
        if self.model is None:
            raise ValueError("No trained model to save. Please train the model first.")
        
        model_data = {
            "model": self.model,
            "transformer": self.transformer,
            "feature_names": self.feature_names,
            "active_categorical_features": self.active_categorical_features,
            "categorical_features": self.categorical_features,
            "adjusted_column": self.adjusted_col,
            "random_state": self.random_state,
            "study": self.study,
            "best_params": self.best_params
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.transformer = model_data["transformer"]
        self.feature_names = model_data["feature_names"]
        self.active_categorical_features = model_data["active_categorical_features"]
        self.categorical_features = model_data["categorical_features"]
        self.adjusted_col = model_data["adjusted_column"]
        self.random_state = model_data["random_state"]
        self.study = model_data.get("study")
        self.best_params = model_data.get("best_params")
        
        print(f"Model loaded from {filepath}")

    @classmethod
    def from_saved_model(cls, filepath: str) -> "RandomForestAdjusted":
        """
        Create a new instance from a saved model
        """
        instance = cls()
        instance.load_model(filepath)
        return instance
    
    def predict(self, X: pd.DataFrame, prediction_column: str = "predicted_points") -> pd.DataFrame:
        """
        Predict using the trained Random Forest model

        Parameters:
            X: Input DataFrame
            prediction_column: Name of the column the predictions will be inserted
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call the train of load_model method first.")
        
        if self.transformer is None:
            raise ValueError("Transformer not found.")
        
        X_copy = X.copy()

        X_transformed, _, _, _ = transform_features(
            data=X_copy, features=self.feature_names, categorical_features=self.categorical_features,
            transformer=self.transformer
        )

        X_selected = X_transformed[self.feature_names]

        # Check for missing values
        missing_mask = X_selected.isnull().any(axis=1)
        
        if missing_mask.any():
            print(f"Warning: {missing_mask.sum()} rows have missing values in required features")
            print("These rows will have NaN predictions")

        predictions = np.full(len(X_selected), np.nan)  # Initialize with NaN
        valid_indices = ~missing_mask

        if valid_indices.any():
            preds = self.model.predict(X_selected[valid_indices])
            predictions[valid_indices] = preds
        
        X_copy[prediction_column] = predictions

        return X_copy
    
    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from the trained Random Forest model and plot top features
        
        Parameters:
            top_n: Number of top features to display in plot
            
        Returns:
            feature_inp: DataFrame with feature importance sorted by importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please call the train method first.")
        
        # Random Forest only has feature_importances_ attribute (Gini importance)
        importance = self.model.feature_importances_
        
        feature_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values(by="importance", ascending=False)
        
        # Plot top 15 features
        top_features = feature_imp.head(top_n)
        
        plt.figure(figsize=(12, 6), facecolor="none")
        plt.barh(range(len(top_features)), top_features["importance"], color="darkslateblue", alpha=0.7)
        plt.yticks(range(len(top_features)), top_features["feature"])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Most Important Features")
        plt.gca().invert_yaxis()
        plt.gca().set_facecolor("none")
        plt.grid(axis="x", alpha=0.4)
        plt.tight_layout()
        plt.show()

        return feature_imp
    