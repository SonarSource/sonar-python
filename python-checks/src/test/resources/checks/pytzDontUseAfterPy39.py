from datetime import datetime

def on_import():
    import pytz # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
          #^^^^

    from pytz import timezone # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
        #^^^^

    import pytz as p # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
          #^^^^
