from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib

def mlp(X_train,y_train) :

    mlp = MLPRegressor( hidden_layer_sizes=3, max_iter=200,
                       random_state=1)
    mlp.fit(X_train, y_train)
    joblib.dump(mlp, 'mlp.pkl')
    return mlp