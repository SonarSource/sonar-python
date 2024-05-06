from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV

def basic():
    pipe = Pipeline(steps=[("clf", SVC())])

    pipe.set_params(clf__F=10, clf__C=45) # Noncompliant {{Provide valid parameters to the estimator.}}
    #               ^^^^^^

def nested_pipelines():

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    p1 = Pipeline(steps=[("scaler", StandardScaler()), ("decomp", PCA())])
    p2 = Pipeline(steps=[("classif", SVC())])

    p = Pipeline(steps=[("preprocess", p1), ("clf", p2)])
    p.set_params(preprocess__scaler__dfgpkdfgs=5) # Noncompliant {{Provide valid parameters to the estimator.}}
    #            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def deeper_nested_pipelines():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    p1_1 = Pipeline(steps=[("scaler", StandardScaler()), ("decomp", PCA())])
    p1_2 = Pipeline(steps=[("classif", SVC())])

    p1 = Pipeline(steps=[("p1_1", p1_1), ("p1_2", p1_2)])

    p2 = Pipeline(steps=[("p1", p1)])

    p3 = Pipeline(steps=[("p2", p2)])
    p4 = Pipeline(steps=[("p3", p3)])
    p4.set_params(p3__p2__p1__p1_1__scaler__dfgpkdfgs=5) # Noncompliant {{Provide valid parameters to the estimator.}}

def deeper_nested_pipelines2():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    p1_1 = Pipeline(steps=[("scaler", StandardScaler()), ("decomp", PCA())])
    p1_2 = Pipeline(steps=[("classif", SVC())])

    p1 = Pipeline(steps=[(not_a_string(), p1_1), ("p1_2", p1_2)])

    p2 = Pipeline(steps=[("p1", p1)])

    p3 = Pipeline(steps=[("p2", p2)])
    p4 = Pipeline(steps=[("p3", p3)])
    p4.set_params(p3__p2__p1__p1_1__scaler__dfgpkdfgs=5)


def recursive_nested_pipelines():
    p1, p2 = Pipeline(steps=[("P2", p2)]), Pipeline(steps=[("P1", p1)])
    p1.set_params(P2__P1__C=45)

def param_grid_dict():
    pipe = Pipeline(steps=[("clf", SVC())])
    param_grid = {
        'clf__G': [0.1, 1, 10], # Noncompliant {{Provide valid parameters to the estimator.}}
    #   ^^^^^^^^
        'clf__kernel': ['linear', 'rbf']
    }
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)

def param_grid_dict2():
    pipe = Pipeline(steps=[("clf", SVC())])
    param_grid = {
        'clf__G': [0.1, 1, 10], # Noncompliant {{Provide valid parameters to the estimator.}}
    #   ^^^^^^^^
        'clf__kernel': ['linear', 'rbf']
    }
    grid = HalvingGridSearchCV(pipe, param_grid=param_grid, cv=5)

def incorrect_pipelines():
    p1 = Pipeline(steps=[("clf", SVC(), "a")])
    p1.set_params(clf__C=45)

    p2 = Pipeline(steps=[("clf", SVC())])
    pgrid2 = {
        'clf': [0.1, 1, 10],
        'clf__G__': [0.1, 1, 10]
    }
    grid2 = GridSearchCV(p2, param_grid=pgrid2)

def ignored_pipelines():
    p1 = Pipeline(steps=[something(), SVC()])
    p1.set_params(clf__C=45)

    p2 = Pipeline(steps=["classif", FantasyClassifier()])
    p2.set_params(classif__C=45)

    p3 = Pipeline(steps=["classif", FantasyClassifier()])
    p3.set_params(C=45)

    p4 = make_pipeline(SVC())
    p4.set_params(clf__D=45)

    p5_1 = make_pipeline(StandardScaler(), PCA())
    p5 = Pipeline(steps=[("preprocess", p5_1), ("clf", SVC())])
    p5.set_params(preprocess__scaler__dfgpkdfgs=5)

