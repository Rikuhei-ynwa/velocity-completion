import time
import numpy as np
import torch
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

import event_config as cfg

from models.utils import loss_reconst

# Rikuhei Umemoto, 2024
# anonymous, 2021
# modifying the code
# https://github.com/ezhan94/multiagent-programmatic-supervision
# https://github.com/keisuke198619/C-OBSO

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES
VEL_FAKE_NODES = cfg.VEL_FAKE_NODES
EVENTTYPE_CATEGORIES = cfg.EVENT_TYPE


class RandomForest:
    """Random Forest model for velocity prediction."""

    def __init__(self, params, parser=None):
        self.device = "cpu"
        self.params = params
        self.batchsize = params["batchsize"]
        self.n_all_agents = params["n_all_agents"]
        self.n_players = params["n_all_agents"] - 1
        self.n_input = params["n_input"]
        self.n_output = params["n_output"]

        # Random Forest parameters
        self.n_estimators = params.get("n_estimators", 100)
        self.max_depth = params.get("max_depth", None)
        self.min_samples_split = params.get("min_samples_split", 2)
        self.min_samples_leaf = params.get("min_samples_leaf", 1)
        self.random_state = params.get("seed", 0)

        # Initialize MultiOutputRegressor with RandomForestRegressor for all players and velocity components
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
            )
        )

        self.is_fitted = False
        self.training_data = []
        self.training_labels = []
        self.data_collection_complete = False

    def restore(self, data, info):
        """
        players_vel_info = np.array(
            [players_mean_v0, players_std_v0, players_mean_v1, players_std_v1],
            )
        """
        data[:, :, 0] = data[:, :, 0] * info[1] + info[0]
        data[:, :, 1] = data[:, :, 1] * info[3] + info[2]

        return data

    def custom_rmse_score(self, y_true, y_pred):
        y_true = y_true.reshape(-1, self.n_players, 2)
        y_pred = y_pred.reshape(-1, self.n_players, 2)
        error = (y_pred[:, :, 0] - y_true[:, :, 0]) ** 2 + (
            y_pred[:, :, 1] - y_true[:, :, 1]
        ) ** 2
        error = np.sum(error, axis=1) / self.n_players
        error = np.sqrt(error)
        error = np.mean(error)
        return -error  # Negate for maximization

    def fit_with_datasets(
        self, train_loader, val_loader, players_info_train, players_info_val
    ):
        """Fit RF model using training and validation datasets."""
        print("Collecting training data for SVM...")
        start_time = time.time()

        # Collect training data
        train_X, train_y = [], []
        for items in train_loader:
            x_t = torch.cat(
                [
                    items["cies"][:, 0:1, :],
                    items["cies"][:, 2 : len(cfg.LOC_NODES), :],
                ],
                dim=1,
            )
            v_t = items["cies"][
                :, len(cfg.LOC_NODES) : len(cfg.LOC_NODES) + len(cfg.VEL_NODES), :
            ]

            X_batch = x_t.detach().cpu().numpy()
            y_batch = v_t[:, 1:, :].detach().cpu().numpy()  # Exclude ball velocity

            train_X.append(X_batch.reshape(X_batch.shape[0], -1))
            train_y.append(y_batch.reshape(y_batch.shape[0], -1))

        # Collect validation data
        val_X, val_y = [], []
        for items in val_loader:
            x_t = torch.cat(
                [
                    items["cies"][:, 0:1, :],
                    items["cies"][:, 2 : len(cfg.LOC_NODES), :],
                ],
                dim=1,
            )
            v_t = items["cies"][
                :, len(cfg.LOC_NODES) : len(cfg.LOC_NODES) + len(cfg.VEL_NODES), :
            ]
            X_batch = x_t.detach().cpu().numpy()
            y_batch = v_t[:, 1:, :].detach().cpu().numpy()  # Exclude ball velocity

            val_X.append(X_batch.reshape(X_batch.shape[0], -1))
            val_y.append(y_batch.reshape(y_batch.shape[0], -1))

        # Concatenate all data
        X_train = np.vstack(train_X)
        y_train = np.vstack(train_y)
        X_val = np.vstack(val_X)
        y_val = np.vstack(val_y)

        """
        players_vel_info = np.array(
            [players_mean_v0, players_std_v0, players_mean_v1, players_std_v1],
            )
        """
        players_info_train = players_info_train.numpy()
        players_info_val = players_info_val.numpy()
        # Restore original scale
        y_train[:, ::2] = (
            y_train[:, ::2] * players_info_train[1] + players_info_train[0]
        )
        y_train[:, 1::2] = (
            y_train[:, 1::2] * players_info_train[3] + players_info_train[2]
        )
        y_val[:, ::2] = y_val[:, ::2] * players_info_val[1] + players_info_val[0]
        y_val[:, 1::2] = y_val[:, 1::2] * players_info_val[3] + players_info_val[2]

        # Combine train and validation data for grid search cross-validation
        # X_combined = np.vstack([X_train, X_val])
        X_combined = np.vstack([X_train, X_val])
        # y_combined = np.vstack([y_train, y_val])
        y_combined = np.vstack([y_train, y_val])

        print(f"SVM Grid Search with {X_combined.shape[0]} total samples...")
        print(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}")

        # Define parameter grid for Grid Search
        param_grid = {
            "estimator__n_estimators": [50, 100],
            "estimator__max_depth": [None],
            "estimator__min_samples_split": [2],
            "estimator__min_samples_leaf": [1],
        }

        # Create base Random Forest model
        base_rf = RandomForestRegressor()
        base_model = MultiOutputRegressor(base_rf)

        # Create Grid Search with Cross-Validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=make_scorer(self.custom_rmse_score),
            cv=5,  # 5-fold cross-validation
            verbose=4,
            n_jobs=-1,  # Use all available cores
        )

        print("Performing Grid Search Cross-Validation...")
        # Perform grid search on combined data
        grid_search.fit(X_combined, y_combined)

        # Get the best model
        self.model = grid_search.best_estimator_
        self.is_fitted = True

        print("Random Forest Grid Search completed!")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score (negative RMSE): {grid_search.best_score_:.6f}")
        print(f"Time taken for grid search: {time.time() - start_time:.2f} seconds")

        # Evaluate best model on validation data
        val_predictions = self.model.predict(X_val)
        val_rmse = self.custom_rmse_score(y_val, val_predictions)
        val_score = self.model.score(X_val, y_val)

        print(f"Validation RMSE: {val_rmse:.6f}")
        print(f"Validation R² score: {val_score:.4f}")

        # Store additional information
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_

        return val_rmse, val_score

    def test(self, test_loader, players_info_test, n_sample=1):
        """Fit SVM model using test data only."""
        # Collect test data
        print("Collecting test data for Random Forest...")
        test_rmse, test_score = np.zeros(n_sample), np.zeros(n_sample)
        l_efobsos = np.zeros((n_sample, len(test_loader.dataset)))

        for i in range(n_sample):
            test_X, test_y, test_rb = [], [], []
            for items in test_loader:
                x_t = torch.cat(
                    [
                        items["cies"][:, 0:1, :],
                        items["cies"][:, 2 : len(cfg.LOC_NODES), :],
                    ],
                    dim=1,
                )
                v_t = items["cies"][
                    :,
                    len(cfg.LOC_NODES) : len(cfg.LOC_NODES) + len(cfg.VEL_NODES),
                    :,
                ]
                player_vel_rb = items["cies"][
                    :, len(cfg.LOC_NODES + cfg.VEL_NODES) :, :
                ]

                X_batch = x_t.detach().cpu().numpy()
                y_batch = v_t[:, 1:, :].detach().cpu().numpy()  # Exclude ball velocity
                player_vel_rb = player_vel_rb.detach().cpu().numpy()

                test_X.append(X_batch.reshape(X_batch.shape[0], -1))
                test_y.append(y_batch.reshape(y_batch.shape[0], -1))
                test_rb.append(player_vel_rb)

            # Concatenate all data
            X_test = np.vstack(test_X)
            y_test = np.vstack(test_y)
            rb_test = np.vstack(test_rb)

            """
            players_vel_info = np.array(
                [players_mean_v0, players_std_v0, players_mean_v1, players_std_v1],
                )
            """
            players_info_test = players_info_test.numpy()
            # Restore original scale
            y_test[:, ::2] = (
                y_test[:, ::2] * players_info_test[1] + players_info_test[0]
            )
            y_test[:, 1::2] = (
                y_test[:, 1::2] * players_info_test[3] + players_info_test[2]
            )
            print(f"SVM Testing with {X_test.shape[0]} samples...")

            if not self.is_fitted:
                raise ValueError("Model must be fitted before testing")
            test_predictions = self.model.predict(X_test)
            test_rmse[i] = self.custom_rmse_score(y_test, test_predictions) * -1
            test_score[i] = self.model.score(X_test, y_test)
            l_efobso = (
                loss_reconst(
                    torch.tensor(rb_test.reshape(-1, self.n_players, 2)).to(
                        self.device
                    ),
                    torch.tensor(y_test.reshape(-1, self.n_players, 2)).to(self.device),
                    self.n_players,
                    Sum=False,
                )
                .detach()
                .cpu()
                .numpy()
            )
            l_efobsos[i, : l_efobso.shape[0]] = l_efobso

        return test_rmse, test_score, l_efobsos
