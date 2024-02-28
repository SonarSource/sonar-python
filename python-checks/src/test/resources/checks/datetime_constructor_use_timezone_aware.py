def some_function():
    from datetime import datetime

    datetime.utcnow() # Noncompliant

    timestamp = 1571595618.0
    datetime.utcfromtimestamp(timestamp) # Noncompliant

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