from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

pipe = Pipeline(steps=[("clf", SVC())])

pipe.set_params(clf__F=10) # Noncompliant
#               ^^^^^^