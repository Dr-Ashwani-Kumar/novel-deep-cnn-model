from sklearn.svm import SVC

def svc(X_train, y_train, X_test, y_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    model = SVC()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test
