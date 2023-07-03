from sklearn.model_selection import GridSearchCV

from utils.timer import timer_decorator


@timer_decorator
def get_best_params(estimator, X_train, y_train, param_grid, cv=5):
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params
