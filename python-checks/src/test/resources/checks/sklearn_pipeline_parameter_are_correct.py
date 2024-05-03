from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def aa():
    pipe = Pipeline(steps=[("clf", SVC())])

    pipe.set_params(clf__F=10, clf__C=45) # Noncompliant {{Provide valid parameters to the estimator.}}
    #               ^^^^^^

def param_grid_dict():
    pipe = Pipeline(steps=[("clf", SVC())])
    param_grid = {
        'clf__G': [0.1, 1, 10], # Noncompliant {{Provide valid parameters to the estimator.}}
    #   ^^^^^^^^
        'clf__kernel': ['linear', 'rbf']
    }
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)

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
