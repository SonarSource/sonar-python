
unrelated_unknown_function()
def simple_cases():
    from datetime import time
    t = time(16, 0)
    formatted_time1 = t.strftime("%H:%M %p") # Noncompliant {{Use %I (12-hour clock) or %H (24-hour clock) without %p (AM/PM).}}
                                #^^^^^^^^^^
    formatted_time2 = t.strftime("%I:%M") # Noncompliant {{Use %I (12-hour clock) with %p (AM/PM).}}
                                #^^^^^^^
    formatted_time3 = t.strftime("%I:%M %p")
    formatted_time4 = t.strftime("%H:%M")
    formatted_time5 = t.strftime()
    formatted_time6 = t.strftime(True)
    formatted_time7 = t.strftime("No formatting")
    formatted_time8 = t.strftime("%I:%M:%S %p")
    formatted_time9 = t.strftime("%I:%M:%S %p %z")
    formatted_time10 = t.strftime("%I:%M:%S %p %Z")
    formatted_time11 = t.strftime("%I:%M:%S %p %f")
    formatted_time12 = t.strftime("%I:%M:%S %p %j")
    formatted_time13 = t.strftime("%I:%M:%S %p %U")
    formatted_time14 = t.strftime("%I:%M:%S %p %W")
    formatted_time15 = t.strftime("%I:%M:%S %p %c")
    formatted_time16 = t.strftime("%I:%M:%S %p %x")
    formatted_time17 = t.strftime("%I:%M:%S %p %X")
    formatted_time18 = t.strftime("%I:%M:%S %p %G")
    formatted_time19 = t.strftime(" %p %H %M") # Noncompliant {{Use %I (12-hour clock) or %H (24-hour clock) without %p (AM/PM).}}
                                 #^^^^^^^^^^^
    formatted_time20 = t.strftime("Hour :%I , minutes :  %M") # Noncompliant {{Use %I (12-hour clock) with %p (AM/PM).}}
    formatted_time21 = t.strftime("Hour :%H , minutes :  %M , useless : %p") # Noncompliant {{Use %I (12-hour clock) or %H (24-hour clock) without %p (AM/PM).}}
    format1 = "%H:%M %p"
             #^^^^^^^^^^> 1 {{Wrong format created here.}}
    t.strftime(format1) # Noncompliant {{Use %I (12-hour clock) or %H (24-hour clock) without %p (AM/PM).}}
              #^^^^^^^

    if cond():
        format2 = "%I:%M"
    else:
        format2 = "%H:%M"
    t.strftime(format2) # FN because of the limitations of SingleAssignedValue

    # This one is a FN because we don't support f-strings in this rule
    format19 = f"%I:%M"
    t.strftime(format19)

    # This one is a FN because we don't support f-strings in this rule
    format_marker = "%p"
    t.strftime(f"%I:%M {format_marker}")
