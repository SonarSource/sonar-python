def some_function():
    from datetime import datetime

    datetime.utcnow()  # Noncompliant {{Don't use `datetime.datetime.utcnow` to create this datetime object.}}
#   ^^^^^^^^^^^^^^^
    timestamp = 1571595618.0
    datetime.utcfromtimestamp(  # Noncompliant {{Don't use `datetime.datetime.utcfromtimestamp` to create this datetime object.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^
        timestamp
    )


def other_function():
    import datetime

    timestamp = 1571595618.0
    datetime.datetime.utcnow()  # Noncompliant
    datetime.datetime.utcfromtimestamp(timestamp)  # Noncompliant
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    test = "test"
    a = f"test/{datetime.datetime.utcnow().strftime('%Y-%m-%d')}/{str(test)}"  # Noncompliant
#               ^^^^^^^^^^^^^^^^^^^^^^^^

    from datetime import datetime as dt

    dt.utcnow()  # Noncompliant
    dt.utcfromtimestamp(timestamp)  # Noncompliant


def unknown_symbols():
    unrelated_call()

    datetime.datetime.utcnow()  # Noncompliant
    datetime.datetime.utcfromtimestamp()  # Noncompliant


def compliant_examples():
    from datetime import datetime, timezone

    datetime.now(timezone.utc)
    timestamp = 1571595618.0
    datetime.fromtimestamp(timestamp, timezone.utc)


def aliased_utcnow():
    from datetime import datetime

    reassigned = datetime.utcnow
    reassigned()  # Noncompliant

    from datetime import datetime as aliased

    aliased.utcnow()  # Noncompliant
