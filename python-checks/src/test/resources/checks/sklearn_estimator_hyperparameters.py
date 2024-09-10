from sklearn.ensemble import (
        AdaBoostClassifier, AdaBoostRegressor, 
        GradientBoostingClassifier, GradientBoostingRegressor,
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor,
        )
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier,KNeighborsRegressor 
from sklearn.svm import SVC, SVR, NuSVC, NuSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV,HalvingRandomSearchCV, RandomizedSearchCV

def non_compliant():
    AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0) # Noncompliant  {{Add the missing hyperparameter learning_rate for this Scikit-learn estimator.}}
   #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AdaBoostRegressor(random_state=0, n_estimators=100) # Noncompliant
    GradientBoostingClassifier( max_depth=1, random_state=0) # Noncompliant
    GradientBoostingRegressor( max_depth=1, random_state=0) # Noncompliant
    HistGradientBoostingClassifier() # Noncompliant
    HistGradientBoostingRegressor() # Noncompliant
    RandomForestClassifier() # Noncompliant
    RandomForestRegressor(n_estimators=50) # Noncompliant

    ElasticNet(random_state=0) # Noncompliant

    NearestNeighbors(radius=0.4) # Noncompliant
    KNeighborsClassifier() # Noncompliant
    KNeighborsRegressor() # Noncompliant

    SVC() # Noncompliant
    SVC(random_state=42) # Noncompliant
    SVC(C=1) # Noncompliant
    SVR() # Noncompliant {{Add the missing hyperparameters C, kernel and gamma for this Scikit-learn estimator.}}
    SVR(C=1.2, kernel="poly") # Noncompliant
    NuSVC() # Noncompliant
    NuSVR(gamma="scale", kernel="poly") # Noncompliant

    DecisionTreeClassifier() # Noncompliant
    DecisionTreeRegressor() # Noncompliant

    MLPClassifier(random_state=1, max_iter=300) # Noncompliant
    MLPRegressor(activation="relu", random_state=1, max_iter=300) # Noncompliant

    PolynomialFeatures(interaction_only=True) # Noncompliant


    pipe = Pipeline([
        ('svc'), SVC()]) # Noncompliant

    pipe2 = make_pipeline(SVC()) # Noncompliant

def compliant():
    AdaBoostClassifier(learning_rate=1.0, n_estimators=100, algorithm="SAMME", random_state=0)
    AdaBoostRegressor(random_state=0, learning_rate=0.2)
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0) 
    GradientBoostingRegressor( max_depth=1, random_state=0, learning_rate=None)
    HistGradientBoostingClassifier(learning_rate=1)
    HistGradientBoostingRegressor(learning_rate=1)
    RandomForestClassifier(min_samples_leaf=1, max_features="sqrt")
    RandomForestRegressor(n_estimators=50, min_samples_leaf=1, max_features="sqrt")

    ElasticNet(alpha=1.0, random_state=0, l1_ratio=0.2) 
    ElasticNet(1.0, random_state=0, l1_ratio=0.2) 

    NearestNeighbors(n_neighbors=2, radius=0.4)
    KNeighborsClassifier(n_neighbors=3)
    KNeighborsRegressor(2)

    SVC(C=1, gamma="scale", kernel="poly") 
    SVR(C=1, gamma="scale", kernel="poly") 
    NuSVC(nu=1, gamma="scale", kernel="poly") 
    NuSVR(C=1, gamma="scale", kernel="poly") 

    DecisionTreeClassifier(ccp_alpha=0.3) 
    DecisionTreeRegressor(ccp_alpha=0.3) 

    MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)
    MLPClassifier((100,),random_state=1)
    MLPRegressor((100,))

    PolynomialFeatures(degree=3, interaction_only=True)
    PolynomialFeatures(3, interaction_only=True)

    s = SVC() # FN
    s.set_params(C=10) # FN


    pipe = make_pipeline(SVC()) # FN
    grid = GridSearchCV(pipe, param_grid={'svc__C': [1, 10, 100]}) 

    pipe2 = Pipeline([('svc'), SVC()]) # FN

    grid2 = GridSearchCV(pipe2, param_grid={'svc__C': [1, 10, 100]}) 

    GridSearchCV(s, param_grid={'svc__C': [1, 10, 100]}) # FN

    classifier = RandomForestClassifier() # FN
    HalvingRandomSearchCV(classifier, {})  
    HalvingGridSearchCV(classifier, {'svc__C': [1, 10, 100]}) 
    RandomizedSearchCV(SVC(), dict()) # FN
