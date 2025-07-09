from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet

def failure():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y) # Noncompliant {{Provide a seed for the random_state parameter.}}
    #                                  ^^^^^^^^^^^^^^^^
    svc = SVC() # Compliant

    svc2 = SVC(probability=True) # Noncompliant
    #      ^^^

    mlp = MLPClassifier() # Noncompliant

    X, y = make_blobs(n_samples=1300, random_state=None) # Noncompliant
    #      ^^^^^^^^^^

    lr_default = LogisticRegression() # Compliant

    lr_sag = LogisticRegression(solver="sag") # Noncompliant
    #        ^^^^^^^^^^^^^^^^^^

    lr_saga = LogisticRegression(solver='saga') # Noncompliant
    #         ^^^^^^^^^^^^^^^^^^

    lr_unknown_string = LogisticRegression(solver=some_unknown_string) # Compliant

    ridge_default = Ridge() # Compliant

    ridge_sag = Ridge(solver="sag") # Noncompliant
    #           ^^^^^

    ridge_saga = Ridge(solver='saga') # Noncompliant
    #            ^^^^^

    lasso_default = Lasso() # Compliant

    lasso_random = Lasso(selection="random") # Noncompliant
    #              ^^^^^

    elastic_default = ElasticNet() # Compliant

    elastic_random = ElasticNet(selection='random') # Noncompliant
    #                ^^^^^^^^^^

def success():
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(random_state=0) # Compliant
    from sklearn.linear_model import SGDClassifier
    sgd = SGDClassifier(random_state=foo()) # Compliant

    def sklearn_seed(rng):
        svc = SVC(random_state=rng) # Compliant

    def foo(random_state=None):
        pass

    foo() # Compliant

def ambiguous():
    from sklearn.svm import SVC as something
    from sklearn.datasets import make_blobs as something

    something = something() # FN

