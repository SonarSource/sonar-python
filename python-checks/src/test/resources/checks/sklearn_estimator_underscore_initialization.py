from sklearn.base import BaseEstimator, RegressorMixin

class MyEstimator(BaseEstimator):
     #^^^^^^^^^^^> {{The attribute is used in this estimator}}
    def __init__(self) -> None:
        self.a_ = None # Noncompliant {{Move this estimated attribute in the `fit` method.}}
        #    ^^
        self.something_ok = True
        self.__something_private__ = True
        local_variable = 5.
        callable().a_  = []

    def another_method(self):
        self.b_ = True

    def fit(x, y):
        self.a_ = 5

# def CustomRegressor

class UnrelatedClass():
    def __init__(self) -> None:
        self.a_ = None

class InheritingEstimator(MyEstimator):
    def __init__(self) -> None:
        self.a_ = None # Noncompliant
        #    ^^

class AnotherUnrelatedClass(UnrelatedClass):
    def __init__(self) -> None:
        self.a_ = None

class DirectlyFromMixin(RegressorMixin):
    def __init__(self) -> None:
        self.a_ = None # Noncompliant

class IndirectlyFromMixinStep1():
    ...
class IndirectlyFromMixinStep2(RegressorMixin):
    ...
class IndirectlyFromMixinStep3(IndirectlyFromMixinStep2):
    ...

class IndirectlyFromMixin(IndirectlyFromMixinStep1, IndirectlyFromMixinStep3):
    def __init__(self) -> None:
        self.a_ = None # Noncompliant

def __init__():
    ...

class Duplicated(BaseEstimator):
    def __init__(self) -> None:
        self.something = None

class Duplicated(BaseEstimator):
    def __init__(self) -> None:
        self.something = True

class TestQuickFix(BaseEstimator):
    def __init__(self):
        self.a_, self.b_ = None, True # Noncompliant
        # Noncompliant@-1
        self.c_ = True # Noncompliant
        (self.d_, self.e_) = None, True # Noncompliant
        # Noncompliant@-1
        self.f_, (self.g_, self.h_) = None, True, None # Noncompliant
        # Noncompliant@-1
        # Noncompliant@-2
