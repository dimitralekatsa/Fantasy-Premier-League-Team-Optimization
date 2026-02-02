# # PROGRESSIVE PREDICTION
# gb = GradientBoosting(categorical_features=categorical_features)

# target_season = "25-26"
# train_data_gw1 = dataset_2[~(dataset_2.season == target_season)]
# gb.train(data=train_data_gw1, study_name="gbm_study", model_name="gbm", features=features_t4_f3_r8_d4, rolling_cols=rolling_cols,
#           difficulty_cols=difficulty_cols, include_test_set=False, trials=500, save_results=False)
# predict_gw_1 = dataset_2[(dataset_2.season == target_season) & (dataset_2.gameweek == 1)]
# gw1_preds = gb.predict(X=predict_gw_1)
# gw1_preds.to_csv(f"{path}Predictions/rmse_gw1_preds.csv", index=False)

# for gw in range(1, 38):
#     train_data = dataset_2[(dataset_2.season != target_season) | (
#         (dataset_2.season == target_season) & (dataset_2.gameweek <= gw)
#     )]

#     gb = GradientBoosting(categorical_features=categorical_features)
#     trials = 300 if gw == 1 else (150 if gw <= 5 else 100)
#     gb.train(data=train_data, features=features_t4_f3_r8_d4, rolling_cols=rolling_cols, difficulty_cols=difficulty_cols,
#              include_test_set=False, trials=trials, study_name=f"gbm_study", model_name=f"gbm_gw{gw+1}", save_results=False)
#     predict_data = dataset_2[(dataset_2.season == target_season) & (dataset_2.gameweek == gw + 1)]
#     gw_preds = gb.predict(predict_data)
#     gw_preds.to_csv(f"{path}Predictions/rmse_gw{gw+1}_preds.csv", index=False)

#     del gb
#     gc.collect()



# import pandas as pd
# path = "C:/Users/lekat/Documents/FPL/"
# from fpl_utils import get_fpl_players_details
# player_details = get_fpl_players_details()
# player_details.set_index("code", inplace=True)
# for i in range(1, 39):
#     dataset = pd.read_csv(f"{path}Predictions/top_mae_rmse/top_mae_rmse_gw{i}_preds.csv")
#     dataset["now_cost"] = dataset["code"].map(player_details["now_cost"])
#     dataset.to_csv(f"{path}Predictions/top_mae_rmse/top_mae_rmse_gw{i}_preds.csv", index=False)



# import pandas as pd
# path = "C:/Users/lekat/Documents/FPL/"
# dfs = {}
# wide_dfs = {}
# for i in range(1, 39):
#     preds = pd.read_csv(f"{path}Predictions/top_mae_rmse/top_mae_rmse_gw{i}_preds.csv")
#     long = handle_double_gameweeks(preds)
#     wide = transform_to_wide_format(long)
#     dfs[i] = long
#     wide_dfs[i] = wide



# for i in range(1, 38):
#     for j in range(1, 5):
#         target_gw = i + j
#         if target_gw in wide_dfs:
#             col = f"Points_GW{target_gw}"
            
#             # Merge only on player_id
#             wide_dfs[i] = pd.merge(wide_dfs[i], wide_dfs[target_gw][["player_id", col]], 
#                                   on="player_id", how="outer")
            
            
#             # Fill missing attributes from target_gw
#             attrs_to_fill = ["position", "now_cost", "status", "team_id", "player_name"]
#             for attr in attrs_to_fill:
#                 if attr in wide_dfs[target_gw].columns:
#                     mask = wide_dfs[i][attr].isna()
#                     if mask.any():
#                         attr_map = wide_dfs[target_gw].set_index("player_id")[attr].to_dict()
#                         wide_dfs[i].loc[mask, attr] = wide_dfs[i].loc[mask, "player_id"].map(attr_map)
#             # wide_dfs[i] = wide_dfs[i].drop(columns=[col])



# wide_preds = pd.concat([wide_preds, pd.DataFrame([{
#     "player_id": 661,
#     "player_name": "Ekitiké",
#     "now_cost": 87,
#     "position": "Forward",
#     "Points_GW6": 0,
#     "Points_GW7": 0,
#     "Points_GW8": 0,
#     "Points_GW9": 0,
#     "Points_GW10": 0,
#     "status": "i",
#     "team": "Liverpool",
#     "team_id": 12,
#     "team_code": 14,
#     "opponent_team": "Crystal Palace",
#     "opponent_team_id": 8,
#     "opponent_team_code": 31,
#     "team_difficulty": 4,
#     "opponent_difficulty": 3,
# }])], ignore_index=True)




