import datetime
import pytz

dt = datetime.datetime(2022, 1, 1, tzinfo=pytz.timezone('US/Eastern'))  # Noncompliant

def timezone_in_a_variable():
    some_timezone = pytz.timezone('US/Eastern')
    dt = datetime.datetime(2022, 1, 1, tzinfo=some_timezone)  # Noncompliant
    return dt

some_timezone_1 = pytz.timezone('US/Eastern')
dt1 = datetime.datetime(2022, 1, 1, tzinfo=some_timezone_1)  # FN because of ReachingDefinitionsAnalysis limitations

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
    dt9 = datetime.datetime(*date_tuple, pytz.timezone('US/Eastern'))  # Noncompliant
