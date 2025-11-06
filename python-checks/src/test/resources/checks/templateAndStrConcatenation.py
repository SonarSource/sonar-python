# Cases that should trigger issues (template string + regular string)

name = "Alice"
age = 30

result1 = t"Hello {name}" + " , welcome."  # Noncompliant {{Template strings should not be concatenated with regular strings.}}
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         ^^^^^^^^^^^^^^^@-1< {{Template string}}
#                           ^^^^^^^^^^^^^@-2< {{Regular string}}

result2 = "Hello" + t" {name}"  # Noncompliant {{Template strings should not be concatenated with regular strings.}}
#         ^^^^^^^^^^^^^^^^^^^^
#         ^^^^^^^@-1< {{Regular string}}
#                   ^^^^^^^^^^@-2< {{Template string}}

result3 = t"Name: {name}" + ", " + "Age: " + t"{age}"  # Noncompliant

result4 = T"Hello {name}" + " world"  # Noncompliant
result5 = tr"Hello {name}" + " world"  # Noncompliant  
result6 = rt"Hello {name}" + " world"  # Noncompliant

greeting = "Hello" + " world"  # Compliant

full_greeting = t"Hello {name}" + t" you are {age}"  # Compliant

template_only1 = t"Hello {name}" + T" you are {age}"  # Compliant
template_only2 = rt"Hello {name}" + tr" you are {age}"  # Compliant

comparison = t"Hello {name}" == " world"  # Compliant
subtraction = t"Hello {name}" - " world"  # Compliant (would be runtime error, but not the concern of the tested rule)

number_add = 1 + 2  # Compliant
mixed_type = t"Number: {age}" + 42  # Compliant (would be runtime error, but not template string + string)

implicit_concat1 = "Hello" " world"  # Compliant - this is implicit concatenation, not + operator
implicit_concat2 = t"Hello {name}" t" world {age}"  # Compliant - this is implicit concatenation of two template strings

f_string_concat = f"Hello {name}" + "world"  # Compliant - f-strings are not template strings

