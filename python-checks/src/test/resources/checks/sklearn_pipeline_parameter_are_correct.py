from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def aa():
    pipe = Pipeline(steps=[("clf", SVC())])

    pipe.set_params(clf__F=10, clf__C=45) # Noncompliant
    #               ^^^^^^

def incorrect_pipelines():
    p1 = Pipeline(steps=[("clf", SVC(), "a")])
    p1.set_params(clf__C=45)

def ignored_pipelines():
    p1 = Pipeline(steps=[something(), SVC()])
    p1.set_params(clf__C=45)

    p2 = Pipeline(steps=["classif", FantasyClassifier()])
    p2.set_params(classif__C=45)

    p3 = Pipeline(steps=["classif", FantasyClassifier()])
    p3.set_params(C=45)