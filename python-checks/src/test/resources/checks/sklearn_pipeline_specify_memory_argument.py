def compliants():
    from sklearn.pipeline import Pipeline, make_pipeline

    pipeline1 = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LinearDiscriminantAnalysis())
    ], memory=None)

    pipeline2 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(), memory=f"some_cache_{var}")

    pipeline3 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(), memory=get_cache_string())

    from sklearn.pipeline import make_pipeline as mkpipe
    pipeline4 = mkpipe(StandardScaler(), LinearDiscriminantAnalysis(), memory=None)
    pipeline5 = mkpipe(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant
    mkpipe(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant
    _ = mkpipe(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant


def non_compliants():
    from sklearn.pipeline import Pipeline, make_pipeline

    pipeline1 = Pipeline([ # Noncompliant {{Specify a memory argument for the pipeline.}}
               #^^^^^^^^
        ('scaler', StandardScaler()),
        ('classifier', LinearDiscriminantAnalysis())
    ])



    pipeline2 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant
               #^^^^^^^^^^^^^


    pipe1 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) # Compliant
    pipe2 = make_pipeline(RobustScaler(), KNeighborsClassifier()) # Compliant

    pipe = Pipeline([ # Noncompliant
          #^^^^^^^^
      ("feature_engineering", PolynomialFeatures(2)),
      ("clf", VotingClassifier([pipe1, pipe2]))
    ])
    Pipeline([ # Noncompliant
   #^^^^^^^^
          ("feature_engineering", PolynomialFeatures(2)),
          ("clf", VotingClassifier([pipe1, pipe2]))
        ])

    a, b = Pipeline([ # Noncompliant
          #^^^^^^^^
                 ("feature_engineering", PolynomialFeatures(2)),
                 ("clf", VotingClassifier([pipe1, pipe2]))
               ]), True

    something.variable = Pipeline([ # Noncompliant
                        #^^^^^^^^
             ("feature_engineering", PolynomialFeatures(2)),
             ("clf", VotingClassifier([pipe1, pipe2]))
           ])

    (e, f), b = True, True,  Pipeline([ # Noncompliant
                 ("feature_engineering", PolynomialFeatures(2)),
                 ("clf", VotingClassifier([pipe1, pipe2]))
               ])

def other():
    from sklearn.pipeline import Pipeline, make_pipeline

    pipeline1 = Pipeline([ # Noncompliant
        ('scaler', StandardScaler()),
        ('classifier', LinearDiscriminantAnalysis())
    ])

    if another_call(some_call(pipeline1)):
        ...

    pipeline2 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant
    if pipeline2 == None:
        ...

    pipeline3 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant
    def a_call(): ...
    def b_call(): ...
    a_call(b_call(pipeline3))