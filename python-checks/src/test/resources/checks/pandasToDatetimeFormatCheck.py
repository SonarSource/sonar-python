import pandas as pd


def yearfirst_true_noncompliant():
    pd.to_datetime("02-03-2022", yearfirst=True) # Noncompliant {{Remove this `yearfirst=True` parameter or make sure the provided date(s) can be parsed accordingly.}}
#                  ^^^^^^^^^^^^>1^^^^^^^^^^^^^^
    pd.to_datetime(
        "02/03/2022",
    #   ^^^^^^^^^^^^>  {{Invalid date.}}
        yearfirst=True # Noncompliant
    ) # ^^^^^^^^^^^^^^

    pd.to_datetime("02 03 2022", yearfirst=True) # Noncompliant
    pd.to_datetime("02.03.2022", yearfirst=True) # Noncompliant
    pd.to_datetime("02;03;2022", yearfirst=True) # Noncompliant
    pd.to_datetime("02_03_2022", yearfirst=True) # Noncompliant


def list_argument_noncompliant():
    pd.to_datetime(["02-03-2022"], yearfirst=True) # Noncompliant
    pd.to_datetime(
        ["02-03-2022", "03-03-2022"],
    #   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^>1 {{This contains invalid date(s).}}
    #    ^^^^^^^^^^^^@-1>2 {{Invalid date.}}
    yearfirst=True   # Noncompliant
#   ^^^^^^^^^^^^^^ {{Remove this `yearfirst=True` parameter or make sure the provided date(s) can be parsed accordingly.}}
    )
    # Following case will actually raise an exception
    pd.to_datetime(["2022-03-02", "03-03-2022"], yearfirst=True) # Noncompliant
    pd.to_datetime([unknown(), "03-03-2022"], yearfirst=True) # Noncompliant
    some_date = "03-03-2022"
    #           ^^^^^^^^^^^^>2
    pd.to_datetime([unknown(), some_date], yearfirst=True) # Noncompliant
    #              ^^^^^^^^^^^^^^^^^^^^^^>1^^^^^^^^^^^^^^


def dataflow_arguments():
    my_date_01 = "02-03-2022"
#                ^^^^^^^^^^^^>2 {{Invalid date.}}
    pd.to_datetime(
        my_date_01,
#       ^^^^^^^^^^>1  {{This contains invalid date(s).}}
        yearfirst=True  # Noncompliant
#       ^^^^^^^^^^^^^^
    )

    my_date_02 = "02-04-2022"
    my_date_02 = "02-03-2022"
    pd.to_datetime(my_date_02, yearfirst=True)  # FN
    is_yearfirst = True
    pd.to_datetime("02-03-2022", yearfirst=is_yearfirst)  # Noncompliant

    my_date_list = ["2022-03-02", "02-03-2022"]
    pd.to_datetime(my_date_list, yearfirst=True)  # Noncompliant

    my_unknown_date = unknown()
    pd.to_datetime(my_unknown_date, yearfirst=True) # OK
    pd.to_datetime(unknown_call(), yearfirst=True) # OK
    pd.to_datetime([unknown_call()], yearfirst=True) # OK
    pd.to_datetime([my_unknown_date, "20-03-2022"], yearfirst=True) # Noncompliant

    my_date_literal = "01-01-2022"
#                     ^^^^^^^^^^^^>
    my_list_literal = [my_date_literal]
    pd.to_datetime(my_list_literal, yearfirst=True)  # Noncompliant
#                  ^^^^^^^^^^^^^^^> ^^^^^^^^^^^^^^


def yearfirst_true_compliant():
    pd.to_datetime("2022-02-03", yearfirst=True)
    pd.to_datetime("2022/02/03", yearfirst=True)
    pd.to_datetime("2022 02 03", yearfirst=True)
    pd.to_datetime("2022.02.03", yearfirst=True)
    pd.to_datetime("2022;02;03", yearfirst=True)
    pd.to_datetime("2022_02_03", yearfirst=True)
    pd.to_datetime("2022 Feb 3", yearfirst=True)
    pd.to_datetime("2022 February 3", yearfirst=True)
    pd.to_datetime("February 3 2022", yearfirst=True) # Can't be parsed: OK by default
    pd.to_datetime("February 3 22", yearfirst=True) # Can't be parsed: OK by default


def yearfirst_false_noncompliant():
    pd.to_datetime("2022-02-03", yearfirst=False)  # Noncompliant
    pd.to_datetime("2022/02/03", yearfirst=False)  # Noncompliant
    pd.to_datetime("2022 02 03", yearfirst=False)  # Noncompliant
    pd.to_datetime("2022.02.03", yearfirst=False)  # Noncompliant
    pd.to_datetime("2022;02;03", yearfirst=False)  # Noncompliant
    pd.to_datetime("2022_02_03", yearfirst=False)  # Noncompliant
    pd.to_datetime("2002-20-02", yearfirst=False)  # Noncompliant
    pd.to_datetime("2022 Feb 3", yearfirst=False)  # Can't be parsed: OK by default
    pd.to_datetime("2022 February 3", yearfirst=True)  # Can't be parsed: OK by default


def yearfirst_false_compliant():
    pd.to_datetime("02-03-2022", yearfirst=False) # OK
    pd.to_datetime("02-03-2022", yearfirst=False, dayfirst=True) # OK
    pd.to_datetime("02-03-2022", yearfirst=False, dayfirst=False) # OK
    pd.to_datetime("02/03/2022", yearfirst=False) # OK
    pd.to_datetime("02 03 2022", yearfirst=False) # OK
    pd.to_datetime("02.03.2022", yearfirst=False) # OK
    pd.to_datetime("02;03;2022", yearfirst=False) # OK
    pd.to_datetime("02_03_2022", yearfirst=False) # OK
    pd.to_datetime("02032022", yearfirst=False) # OK


def dayfirst_true_noncompliant():
    pd.to_datetime("01-22-2000", dayfirst=True)  # Noncompliant
    pd.to_datetime("2000-02-23", dayfirst=True)  # Noncompliant
    pd.to_datetime("2000-02-23", dayfirst=True, yearfirst=True)  # Noncompliant


def dayfirst_false_noncompliant():
    pd.to_datetime("2002-20-02", dayfirst=False)  # Noncompliant
    pd.to_datetime("20-02-2002", dayfirst=False)  # Noncompliant


def dayfirst_false_compliant():
    pd.to_datetime("2002-02-20", dayfirst=False)
    pd.to_datetime("02-20-2002", dayfirst=False)


def datetimes_noncompliant():
    pd.to_datetime(["01-22-2000 10:00"], dayfirst=True)  # Noncompliant {{Remove this `dayfirst=True` parameter or make sure the provided date(s) can be parsed accordingly.}}
    pd.to_datetime(["01-22-2000 10:00"], yearfirst=True)  # Noncompliant {{Remove this `yearfirst=True` parameter or make sure the provided date(s) can be parsed accordingly.}}
    pd.to_datetime(["01-22-2000T10:00"], yearfirst=True)  # Noncompliant
    pd.to_datetime(["01-22-2000 10:00:42"], yearfirst=True)  # Noncompliant
    pd.to_datetime(["01-22-200010:00:42"], yearfirst=True)  # Noncompliant
    pd.to_datetime(["01-22-2000T10:00:42"], yearfirst=True)  # Noncompliant


def datetimes_compliant():
    pd.to_datetime(["2000-03-04 10:00"], dayfirst=True)  # OK
    pd.to_datetime(["2000-03-04 10:00"], yearfirst=True)  # OK
    pd.to_datetime(["01-22-2000T10:00:42.234"], yearfirst=True)  # OK, not parsing milliseconds
    pd.to_datetime(["01-22-2000T10:00:42+02:00"], yearfirst=True)  # OK, not parsing timezones


def various_separators_dayfirst_ok():
    pd.to_datetime("02-03-2022", dayfirst=True)
    pd.to_datetime("02/03/2022", dayfirst=True)
    pd.to_datetime("02 03 2022", dayfirstt=True)
    pd.to_datetime("02.03.2022", dayfirst=True)
    pd.to_datetime("02;03;2022", dayfirst=True)


def yearfirst_YY():
    pd.to_datetime("22-02-03", yearfirst=True)
    pd.to_datetime("22-02-98", yearfirst=True)  # Noncompliant
    pd.to_datetime("31.12.23", yearfirst=True)  # Suspicious, but ok
    pd.to_datetime("22-98-02", yearfirst=True)  # FN


def other():
    unknown()
    pd.to_datetime()
    pd.to_datetime(yearfirst=True)
    pd.to_datetime("20-02-2022", yearfirst="hello")  # Noncompliant
    pd.to_datetime("20-02-2022", yearfirst=[])  # OK
    pd.DataFrame()
