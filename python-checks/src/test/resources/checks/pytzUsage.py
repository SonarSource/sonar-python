from datetime import datetime

def on_import():
    import pytz # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
          #^^^^
    from pytz import timezone # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
        #^^^^
    from pytz import timezone as pytz_timezone
    from pytz.tzinfo import BaseTzInfo # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
        #^^^^^^^^^^^
    from pip._vendor import pytz # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
        #^^^^^^^^^^^
    import pytz as p # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
          #^^^^
    from something.different.pytz import a_function
    import something.different.pytz as p2
    import pytz as p3 # Noncompliant {{Don't use `pytz` module with Python 3.9 and later.}}
