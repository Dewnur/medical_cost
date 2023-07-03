SAVE_MODELS = False
LOAD_MODELS = True
best_dt_filename = "models/best_dt_model.pkl"
best_rf_filename = "models/best_rf_model.pkl"
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_leaf_nodes': [5, 25, 50, 100],
}
