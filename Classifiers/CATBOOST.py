from catboost import CatBoostClassifier


def CATBOOST_classifier(X_train, y_train, X_test, y_test):
    clf = CatBoostClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return preds, y_test