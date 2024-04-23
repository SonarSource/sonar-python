from sklearn.base import BaseEstimator

class MyEstimator(BaseEstimator):
    def __init__(self) -> None:
        self.a_ = None # Noncompliant
        #    ^^
        self.something_ok = True
        local_variable = 5.
        callable().a_  = []

    def another_method(self):
        self.b_ = True

class UnrelatedClass():
    def __init__(self) -> None:
        self.a_ = None
