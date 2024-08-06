def some_function():
    from datetime import datetime
    datetime.utcnow() # Noncompliant {{Don't use `datetime.datetime.utcnow` to create this datetime object.}}
   #^^^^^^^^^^^^^^^^^
    timestamp = 1571595618.0
    datetime.utcfromtimestamp(timestamp) # Noncompliant {{Don't use `datetime.datetime.utcfromtimestamp` to create this datetime object.}}
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

    datetime.datetime.utcnow()
    datetime.datetime.utcfromtimestamp()

def compliant_examples():
    from datetime import datetime, timezone

    datetime.now(timezone.utc)
    timestamp = 1571595618.0
    datetime.fromtimestamp(timestamp, timezone.utc)

def aliased_utcnow():
    from datetime import datetime
    reassigned = datetime.utcnow
    # FN because we lose the FQN in the assignment
    reassigned()

    from datetime import datetime as aliased
    aliased.utcnow() # Noncompliant
