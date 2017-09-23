import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def xgboost(X_train, y_train):
    model = xgb.XGBRegressor(nthread=4)
    params = {'min_child_weight': [2,3],
                'learning_rate':[0.4,0.5,0.6],
              'max_depth': [3,4]}
    grid = GridSearchCV(model, params,cv=10)
    grid.fit(X_train, y_train)
    joblib.dump(grid, 'xgboostModel.pkl')
    return grid
