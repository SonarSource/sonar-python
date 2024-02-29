def some_function():
    from datetime import datetime
    datetime.utcnow() # Noncompliant {{Using timezone aware "datetime"s should be preferred over using "datetime.datetime.utcnow" and "datetime.datetime.utcfromtimestamp"}}
   #^^^^^^^^^^^^^^^^^
    timestamp = 1571595618.0
    datetime.utcfromtimestamp(timestamp) # Noncompliant {{Using timezone aware "datetime"s should be preferred over using "datetime.datetime.utcnow" and "datetime.datetime.utcfromtimestamp"}}
   #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def other_function():
    import datetime
    timestamp = 1571595618.0
    datetime.datetime.utcnow() # Noncompliant
    datetime.datetime.utcfromtimestamp(timestamp) # Noncompliant

    from datetime import datetime as dt
    dt.utcnow() # Noncompliant
    dt.utcfromtimestamp(timestamp) # Noncompliant

def unknown_symbols():

    unrelated_call()

    datetime.datetime.utcnow() # FN because there is no symbol
    datetime.datetime.utcfromtimestamp() # FN because there is no symbol

def compliant_examples():
    from datetime import datetime, timezone

    datetime.now(timezone.utc)
    timestamp = 1571595618.0
    datetime.fromtimestamp(timestamp, timezone.utc)

def aliased_utcnow():
    from datetime import datetime
    aliased = datetime.utcnow
    # FN because we lose the FQN in the assignment
    aliased()

    from datetime.datetime import utcnow
    utcnow() # Noncompliant
