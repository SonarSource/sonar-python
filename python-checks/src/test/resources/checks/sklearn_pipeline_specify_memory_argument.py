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



def non_compliants():
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.pipeline import make_pipeline as mkpipe

    pipeline1 = Pipeline([ # Noncompliant {{Specify a memory argument for the pipeline.}}
               #^^^^^^^^
        ('scaler', StandardScaler()),
        ('classifier', LinearDiscriminantAnalysis())
    ])



    pipeline2 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant
               #^^^^^^^^^^^^^

    pipeline5 = mkpipe(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant
    _ = mkpipe(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant
    mkpipe(StandardScaler(), LinearDiscriminantAnalysis()) # Noncompliant

    pipe1 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()) # Compliant because used in a VotingClassifier
    pipe2 = make_pipeline(RobustScaler(), KNeighborsClassifier()) # Compliant because used in a VotingClassifier

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

def more_nested():
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler

    p1 = make_pipeline(StandardScaler())
    p2 = make_pipeline(StandardScaler())
    p3 = make_pipeline(StandardScaler())
    p4 = make_pipeline(StandardScaler())

    p5, p6 = make_pipeline(p1, p2), make_pipeline(p3, p4) # Noncompliant
                                   #^^^^^^^^^^^^^

    r1 = make_pipeline(p5) # Noncompliant

def more_nested2():
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler

    p1 = make_pipeline(StandardScaler())
    p2 = make_pipeline(StandardScaler())
    p3 = make_pipeline(StandardScaler())
    p4 = make_pipeline(StandardScaler())

    p5, p6 = *(make_pipeline(p1, p2), make_pipeline(p3, p4)) # Noncompliant
                                     #^^^^^^^^^^^^^

    r1 = make_pipeline(p5) # Noncompliant
def more_nested3():
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler

    p1 = make_pipeline(StandardScaler())
    p2 = make_pipeline(StandardScaler())
    p3 = make_pipeline(StandardScaler())
    p4 = make_pipeline(StandardScaler())

    p5, p6 = *[make_pipeline(p1, p2), make_pipeline(p3, p4)] # Noncompliant
                                     #^^^^^^^^^^^^^

    r1 = make_pipeline(p5) # Noncompliant

def more_nested4():
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import StandardScaler
    class SomeClass:
        def __init__(self):
            self.p1 = make_pipeline(StandardScaler())
            p2 = make_pipeline(StandardScaler())
            self.p3 = make_pipeline(StandardScaler())
            p4 = make_pipeline(StandardScaler())

            p5, self.p6 = *[make_pipeline(self.p1, p2), make_pipeline(self.p3, p4)] # Noncompliant
                                                       #^^^^^^^^^^^^^
            r1 = make_pipeline(p5) # Noncompliant


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