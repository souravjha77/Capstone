from sklearn.externals import joblib
from sklearn.svm import SVR


def svr(X_train,y_train,K) :

    regressor = SVR(kernel='rbf')
    regressor.fit(X_train.values,y_train.values)

    # now you can save it to a file
    joblib.dump(regressor, 'svr.pkl')
    return regressor


