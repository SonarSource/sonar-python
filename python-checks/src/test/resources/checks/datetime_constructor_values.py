import datetime

def compliant_examples():
    datetime.time(12, 30, 0, 0)
    datetime.date(2024, 3, 1)
    datetime.datetime(2024, 3, 1, 12, 30, 0, 0)

def non_compliant_examples():
    datetime.date(2024, 3, 32) # Noncompliant {{The day parameter must be valid.}}
                          #^^
    datetime.date(2024, 13, 1) # Noncompliant {{The month parameter must be valid.}}
                       #^^
    datetime.date(2024, 0, 1) # Noncompliant
    datetime.date(2024, 3, 0) # Noncompliant
    datetime.date(-5, 3, 1) # Noncompliant {{The year parameter must be valid.}}
                 #^^
    datetime.date(100000, 5, 3) # Noncompliant

    datetime.time(-5, 30, 0, 0) # Noncompliant {{The hour parameter must be valid.}}
                 #^^
    datetime.time(12, 60, 0, 0) # Noncompliant {{The minute parameter must be valid.}}
                     #^^
    datetime.time(12, 30, 60, 0) # Noncompliant {{The second parameter must be valid.}}
                         #^^
    datetime.time(12, 30, 0, -1) # Noncompliant {{The microsecond parameter must be valid.}}
                            #^^
    datetime.time(12, 30, 0, 1000000) # Noncompliant
    datetime.time(28) # Noncompliant
    datetime.time(5, 61) # Noncompliant
    datetime.time(5, 5, 61) # Noncompliant

    datetime.datetime(2024, 3, 1, 12, 30, -2, 1) # Noncompliant {{The second parameter must be valid.}}
                                         #^^
    datetime.datetime(2024, 3, 1, 12, 30, 0, -1) # Noncompliant {{The microsecond parameter must be valid.}}
                                            #^^
    datetime.datetime(2024, 3, 1, 12, 30, 0, 1000000) # Noncompliant
    datetime.datetime(2024, 3, 32, 12, 30, 0, 0) # Noncompliant {{The day parameter must be valid.}}
                              #^^
    datetime.datetime(2024, 13, 1, 12, 30, 0, 0) # Noncompliant {{The month parameter must be valid.}}
                           #^^
    datetime.datetime(2024, 0, 1, 12, 30, 0, 0) # Noncompliant
    datetime.datetime(-5, 3, 1, 12, 30, 0, 0) # Noncompliant {{The year parameter must be valid.}}
                     #^^
    datetime.datetime(2024, 3, 1, 26, 30, 0, 4) # Noncompliant {{The hour parameter must be valid.}}
                                 #^^
    datetime.datetime(100000, 5, 3, 12, 30, 0, 0) # Noncompliant
    datetime.datetime(2024, 3, 1, 12, 60, 0, 0) # Noncompliant
    datetime.datetime(2024, 3, 1, 12, 30, 60, 0) # Noncompliant
    datetime.datetime(2024, 3, 1, 12, 30, 0, -1) # Noncompliant

def false_negatives_tuple_unpacking():
    datetime.date(*(2024, -1, -1))

def different_imported_name():
    import datetime as dt
    dt.date(2024, 3, 32) # Noncompliant
    dt.time(12, 60, 0, 0) # Noncompliant
    dt.datetime(2024, 3, 32, 12, 30, 0, 0) # Noncompliant

    from datetime import date as d
    from datetime import time as t
    from datetime import datetime as dtt
    d(2024, 3, 32) # Noncompliant
    t(12, 60, 0, 0) # Noncompliant
    dtt(2024, 3, 32, 12, 30, 0, 0) # Noncompliant

def no_issue_on_syntax_errors():
    datetime.time(12, 5, True)
    datetime.date(2024, True, 5)
    datetime.date()
    datetime.time()
    datetime.datetime()
    datetime.date(19)
    datetime.date(19, 5)
    datetime.time()

    datetime.datetime(2024, 3, 1, 12, 30, 0, True)
    datetime.datetime(2024, 3, 1, 12, 30, True, 0)
    datetime.datetime(2024, 3, 1, 12, True, 0, 0)
    datetime.datetime(2024, 3, 1, True, 30, 0, 0)
    datetime.datetime(2024, 3, True, 12, 30, 0, 0)
    datetime.datetime(2024, True, 1, 12, 30, 0, 0)
    datetime.datetime(True, 3, 1, 12, 30, 0, 0)

    datetime.time(-random(), 12, 5, 0)