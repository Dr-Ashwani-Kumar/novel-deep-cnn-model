from xgboost import XGBClassifier


def Xgboost(X_train, y_train, X_test, y_test):
    model = XGBClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test