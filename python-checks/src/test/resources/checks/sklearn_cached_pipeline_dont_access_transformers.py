from sklearn.datasets import load_diabetes
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import FastICA


def case1():
    diabetes = load_diabetes()
    scaler = RobustScaler()
    knn = KNeighborsRegressor(n_neighbors=5)

    pipeline = Pipeline([
              #^^^^^^^^> {{Pipeline created here}}
        ('scaler', scaler), # Noncompliant {{Avoid accessing transformers in a cached pipeline.}}
                  #^^^^^^
        ('knn', knn)
    ], memory="cache")

    print(scaler.center_)
         #^^^^^^< {{Accessed here}}

def case2():
    diabetes = load_diabetes()
    scaler = RobustScaler()
    knn = KNeighborsRegressor(n_neighbors=5)

    pipeline = make_pipeline(scaler, knn, memory=something()) # Noncompliant
                            #^^^^^^

    print(scaler.center_)
         #^^^^^^< {{Accessed here}}

    scaler1, knn1 = RobustScaler(), KNeighborsRegressor(n_neighbors=12)
    p3 = make_pipeline(scaler1, knn1, memory=None)
    p4 = make_pipeline(scaler1, knn1)

    p5 = make_pipeline(smth_unknown, memory="cache")

def case3():
    diabetes = load_diabetes()
    poly = PolynomialFeatures()
    scaler = RobustScaler()
    knn = KNeighborsRegressor(n_neighbors=5)

    pipeline = Pipeline([
              #^^^^^^^^> {{Pipeline created here}}
        ('scaler', scaler), # Noncompliant
                  #^^^^^^
        ('knn', knn),
        ('poly', poly, aaa),
    ], memory="cache")

    print(scaler.center_)
         #^^^^^^< {{Accessed here}}