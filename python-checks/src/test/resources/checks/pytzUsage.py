from datetime import datetime

def on_import():
    import pytz # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
          #^^^^
    from pytz import timezone # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
        #^^^^
    import pytz as p # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
          #^^^^
    from something.different.pytz import a_function
    import something.different.pytz as p2
    import pytz as p3 # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
