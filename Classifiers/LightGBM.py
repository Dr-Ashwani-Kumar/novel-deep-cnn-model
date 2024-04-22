import lightgbm as lgb

def lightgbm(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds, y_test
