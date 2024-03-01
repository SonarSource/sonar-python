import datetime
import pytz

dt = datetime.datetime(2022, 1, 1, tzinfo=pytz.timezone('US/Eastern'))  # Noncompliant {{Don't pass a "pytz.timezone" to the "datetime.datetime" constructor.}}
                                  #^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
dt_ = datetime.datetime(2024, 2, 4, 5, 5, 5, 1, pytz.timezone('US/Eastern'))  # Noncompliant {{Don't pass a "pytz.timezone" to the "datetime.datetime" constructor.}}
                                               #^^^^^^^^^^^^^^^^^^^^^^^^^^^
def timezone_in_a_variable():
    some_timezone = pytz.timezone('US/Eastern')
                   #^^^^^^^^^^^^^^^^^^^^^^^^^^^> 1 {{The pytz.timezone is created here.}}
    dt1 = datetime.datetime(2022, 1, 1, tzinfo=some_timezone)  # Noncompliant {{Don't pass a "pytz.timezone" to the "datetime.datetime" constructor.}}
                                       #^^^^^^^^^^^^^^^^^^^^
    return dt1

some_timezone_1 = pytz.timezone('US/Eastern')
dt1 = datetime.datetime(2022, 1, 1, tzinfo=some_timezone_1)  # FN because of ReachingDefinitionsAnalysis limitations : it only runs in functions

def compliant():
    other_timezone = datetime.timezone(datetime.timedelta(hours=2))
    dt2 = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
    dt3 = datetime.datetime(2022, 1, 1, tzinfo=other_timezone)

    dt4 = datetime.datetime(2022, 1, 1, tzinfo=None)

    dt5 = datetime.datetime(2022, 1, 1)
    def some_function():
        return None
    unrelated_call = some_function()
    unknown_call()
    unknown_call(True)
    unknown_call(True, some_keyword=True)

    dt6 =  datetime.datetime(2022, 1, 1, tzinfo=some_function())
    dt7 =  datetime.datetime(2022, 1, 1, tzinfo=unrelated_call)

    dt8 = datetime.datetime()
def with_tuple_unpacking():
    date_tuple = [2024, 2, 28]
    some_timezone_2 = pytz.timezone('US/Eastern')
    dt9 = datetime.datetime(*date_tuple, tzinfo=some_timezone_2)  # Noncompliant
    dt9_1 = datetime.datetime(*date_tuple, tzinfo=pytz.timezone('US/Eastern'))  # Noncompliant
    full_date_tuple = [2024, 2, 28, 5, 5, 5, 1]
    dt9_2 = datetime.datetime(*full_date_tuple, some_timezone_2)  # FN because nthArgumentOrKeyword does not take into account the size of the tuple

def multiple_definitions():
    if random():
        some_timezone_3 = pytz.timezone('US/Eastern')
                         #^^^^^^^^^^^^^^^^^^^^^^^^^^^> 1 {{The pytz.timezone is created here.}}
    else:
        some_timezone_3 = pytz.timezone('US/Pacific')
                         #^^^^^^^^^^^^^^^^^^^^^^^^^^^> 2 {{The pytz.timezone is created here.}}
    dt10 = datetime.datetime(2022, 1, 1, tzinfo=some_timezone_3)  # Noncompliant {{Don't pass a "pytz.timezone" to the "datetime.datetime" constructor.}}
                                        #^^^^^^^^^^^^^^^^^^^^^^

def with_valid_timezone_alternative():
    import datetime
    class ValidTimeZone(datetime.tzinfo):
        ...

    cond = True
    if cond:
        some_timezone = ValidTimeZone()
    else:
        some_timezone = pytz.timezone('US/Eastern')
                       #^^^^^^^^^^^^^^^^^^^^^^^^^^^> 1 {{The pytz.timezone is created here.}}
    if cond: # This is technically wrong because we will never use the pytz.timezone
        dt11 = datetime.datetime(2022, 1, 1, tzinfo=some_timezone) # Noncompliant {{Don't pass a "pytz.timezone" to the "datetime.datetime" constructor.}}
                                            #^^^^^^^^^^^^^^^^^^^^

    another_timezone = ValidTimeZone()
    aliased_timezone = another_timezone
    another_timezone = pytz.timezone('US/Eastern')
    dt12 = datetime.datetime(2022, 1, 1, tzinfo=aliased_timezone)
    dt13 = datetime.datetime(2022, 1, 1, tzinfo=another_timezone) # Noncompliant {{Don't pass a "pytz.timezone" to the "datetime.datetime" constructor.}}
