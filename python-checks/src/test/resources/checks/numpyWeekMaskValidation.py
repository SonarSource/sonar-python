import numpy as np
from something import some_function as my_function
def unrelated():
    unkwown_function()
    np.busday_offset('2012-05' , 1, roll='forward')
    def other_function(): ...
    other_function()
    np.busday_offset('2012-05' , 1, roll='forward', weekmask=other_function())
    my_function()

def some_cases():
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask='0111100')
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask=[0, 1, 1, 1, 1, 0, 0])
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask="Tue Wed Thu Fri")
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask="Tue Wed Thu Fri Fri")
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask="TueWedThuFriFri")
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask="TueWedThuFri Fri")
    offset = np.busday_offset('2012-05', 1, weekmask="TueWed ThuFri Fri")
    offset = np.busday_offset('2012-05', 1, 'forward', "TueWed ThuFri Fri")

    offset = np.busday_offset('2012-05', 1, 'forward', "TueWed ThuFri igpifdjpigdg") # Noncompliant {{String must be either 7 characters long and contain only 0 and 1, or contain abbreviated weekdays.}}
                                                      #^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask='01')  # Noncompliant {{String must be either 7 characters long and contain only 0 and 1, or contain abbreviated weekdays.}}
                                                                    #^^^^
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask="fsfdiopj") # Noncompliant {{String must be either 7 characters long and contain only 0 and 1, or contain abbreviated weekdays.}}
                                                                    #^^^^^^^^^^
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask=[1,1,1,1,1,1,1,1]) # Noncompliant {{Array must have 7 elements, all of which are 0 or 1.}}
                                                                    #^^^^^^^^^^^^^^^^^
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask=[1,1]) # Noncompliant
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask=["a", "b", "c"]) # Noncompliant {{Array must have 7 elements, all of which are 0 or 1.}}
                                                                    #^^^^^^^^^^^^^^^
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask=[1,1,1,"1",1,1,1]) # Noncompliant {{Array must have 7 elements, all of which are 0 or 1.}}
                                                                    #^^^^^^^^^^^^^^^^^
    offset = np.busday_offset('2012-05', 1, roll='forward', weekmask=[1,1,0,"0",1,1,1]) # Noncompliant {{Array must have 7 elements, all of which are 0 or 1.}}
                                                                    #^^^^^^^^^^^^^^^^^
def with_assigned_values():
    weekmask1 = "Tue Wed Thu Fri"
    offset1 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask1)

    weekmask2 = "0111100"
    offset2 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask2)

    weekmask3 = [0, 1, 1, 1, 1, 0, 0]
    offset3 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask3)

    weekmask4 = "sdfgsdfgsdg"
    offset4 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask4) # Noncompliant

    weekmask5 = [1,1,1,1,1,1,1,1]
               #^^^^^^^^^^^^^^^^^> 1 {{Invalid mask is created here.}}
    offset5 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask5) # Noncompliant {{Array must have 7 elements, all of which are 0 or 1.}}
                                                                     #^^^^^^^^^
    weekmask6 = [1,1]
    offset6 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask6) # Noncompliant

    weekmask7 = ["a", "b", "c"]
    offset7 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask7) # Noncompliant

    weekmask8 = "01"
               #^^^^> 1 {{Invalid mask is created here.}}
    offset8 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask8) # Noncompliant {{String must be either 7 characters long and contain only 0 and 1, or contain abbreviated weekdays.}}
                                                                     #^^^^^^^^^
    weekmask9 = ("TueWed")
    offset9 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask9)

    weekmask10 = [1,1,1,1,1,1,2]
    offset10 = np.busday_offset('2012-05', 1, roll='forward', weekmask=weekmask10) # Noncompliant {{Array must have 7 elements, all of which are 0 or 1.}}

def busday_count():
    offset = np.busday_count('2012-05', '2012-06', weekmask="TueWed ThuFri Fri")
    offset = np.busday_count('2012-05', '2012-06', weekmask="TueWed ThuFri igpifdjpigdg") # Noncompliant
    offset = np.busday_count('2012-05', '2012-06', weekmask='01')  # Noncompliant
    offset = np.busday_count('2012-05', '2012-06', weekmask="fsfdiopj") # Noncompliant
    offset = np.busday_count('2012-05', '2012-06', weekmask=[1,1,1,1,1,1,1,1]) # Noncompliant
    offset = np.busday_count('2012-05', '2012-06', weekmask=[1,1,1,1,1,1,1])
    offset = np.busday_count('2012-05', '2012-06', weekmask=[1,1]) # Noncompliant
    offset = np.busday_count('2012-05', '2012-06', weekmask=["a", "b", "c"]) # Noncompliant
    offset = np.busday_count('2012-05', '2012-06', weekmask=[1,1,1,1,1,1,2]) # Noncompliant
    offset = np.busday_count('2012-05', '2012-06', weekmask=[1,1,1,1,0,1,1,1,1]) # Noncompliant
    offset = np.busday_count('2012-05', '2012-06', weekmask="Tue Wed Thu Fri")
    offset = np.busday_count('2012-05', '2012-06', weekmask="Tue Wed Thu Fri Fri")
    offset = np.busday_count('2012-05', '2012-06', weekmask="TueWedThuFriFri", holidays=["2012-05-01", "2012-05-02"])

def is_busday():
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask="TueWed ThuFri Fri")
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask="TueWed ThuFri igpifdjpigdg") # Noncompliant
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask='01')  # Noncompliant
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask="fsfdiopj") # Noncompliant
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask=[1,1,1,1,1,1,1,1]) # Noncompliant
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask=[1,1,1,0,1,1,1])
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask=[1,1]) # Noncompliant
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask=["a", "b", "c"]) # Noncompliant
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask=[1,1,1,1,1,1,2]) # Noncompliant
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask=[1,1,1,1,0,1,1,1,1]) # Noncompliant
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask="Tue Wed Thu Fri")
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask="Tue Wed Thu Fri Fri")
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], weekmask="TueWedThuFriFri")
    bools = np.is_busday(['2012-05', '2012-06', '2012-07'], holidays=["2012-05-01", "2012-05-02"], weekmask="TueWedThuFriFri")

def busdaycalendar():
    np.busdaycalendar(weekmask="TueWed ThuFri Fri")
    np.busdaycalendar(weekmask="TueWed ThuFri igpifdjpigdg") # Noncompliant
    np.busdaycalendar(weekmask='01')  # Noncompliant
    np.busdaycalendar(weekmask="fsfdiopj") # Noncompliant
    np.busdaycalendar(weekmask=[1,1,1,1,1,1,1,1]) # Noncompliant
    np.busdaycalendar(weekmask=[1,1,1,0,1,1,1])
    np.busdaycalendar(weekmask=[1,1]) # Noncompliant
    np.busdaycalendar(weekmask=["a", "b", "c"]) # Noncompliant
    np.busdaycalendar(weekmask=[1,1,1,1,1,1,2]) # Noncompliant
    np.busdaycalendar(weekmask=[1,1,1,1,0,1,1,1,1]) # Noncompliant
    np.busdaycalendar(weekmask="Tue Wed Thu Fri")
    np.busdaycalendar(weekmask="Tue Wed Thu Fri Fri", holidays=["2012-05-01", "2012-05-02"])
    np.busdaycalendar(holidays=["2012-05-01", "2012-05-02"], weekmask="Tue Wed Thu Fri Fri")

    np.busdaycalendar(weekmask="Sun TueWed  \x0b")
    np.busdaycalendar(weekmask="Sun TueWed  \x0b", holidays=["2012-05-01", "2012-05-02"])
    np.busdaycalendar(weekmask="Sun TueWed  \x0b \t \t \n\r \x0c   Fri")
