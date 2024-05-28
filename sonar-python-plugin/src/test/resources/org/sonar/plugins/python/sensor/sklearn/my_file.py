from sklearn.model_selection import train_test_split

def failure():
    X_train, X_test, y_train, y_test = train_test_split(X, y) # S6709 (Not raised - not using sklearn stubs to analyze sklearn)
