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
              #^^^^^^^^> {{The Pipeline is created here}}
        ('scaler', scaler), # Noncompliant {{Avoid accessing transformers in a cached pipeline.}}
                  #^^^^^^
        ('knn', knn)
    ], memory="cache")

    print(scaler.center_)
         #^^^^^^< {{The transformer is accessed here}}

def case2():
    diabetes = load_diabetes()
    scaler = RobustScaler()
    knn = KNeighborsRegressor(n_neighbors=5)

    pipeline = make_pipeline(scaler, knn, memory=something())

    print(scaler.center_)

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
              #^^^^^^^^> {{The Pipeline is created here}}
        ('scaler', scaler), # Noncompliant
                  #^^^^^^
        ('knn', knn),
        ('poly', poly, aaa),
    ], memory="cache")

    print(scaler.center_)
         #^^^^^^< {{The transformer is accessed here}}

def case4():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    pca = PCA(n_components=2)
    a, b, p1 = True, None, Pipeline([('scaler', scaler), ('pca', pca)], memory="cache") # Noncompliant
    pca.fff_ = 12

def case5():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    pca = PCA(n_components=2)
    Pipeline([('scaler', scaler), ('pca', pca)], memory="cache") # Noncompliant
    pca.fff_ = 12

    scaler2 = StandardScaler()
    pca2 = PCA()
    _, (a, b) = *[True, None, Pipeline([('scaler', scaler), ('pca', pca)], memory="cache")] # Noncompliant
    pca2.fff_ = 12

    scaler3 = StandardScaler()
    pca3 = PCA()
    pipel = Pipeline([(clever_generator())])

def case5():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    class SomeKindOfRecord():
        ...

    obj = SomeKindOfRecord()
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    obj.pipeline =  Pipeline([('scaler', scaler), ('pca', pca)], memory="cache") # Noncompliant

    scaler.szsdgf_ = 12
