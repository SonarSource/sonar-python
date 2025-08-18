def case1():
  iterable = (3, 1, 4, 1)
  list(list(iterable)) # Noncompliant {{Remove this redundant call.}}
#      ^^^^ 
# ^^^^@-1< {{A redundant call is done here.}}
  list(tuple(iterable)) # Noncompliant
  list(sorted(iterable)) # Noncompliant

def case2():
  iterable = (3, 1, 4, 1)
  set(list(iterable)) # Noncompliant
  set(set(iterable)) # Noncompliant
  set(tuple(iterable)) # Noncompliant
  set(reversed(iterable)) # Noncompliant
  set(sorted(iterable)) # Noncompliant

def case3():
  iterable = (3, 1, 4, 1)
  sorted(list(iterable)) # Noncompliant
  sorted(tuple(iterable)) # Noncompliant
  sorted(sorted(iterable)) # Noncompliant

def case4():
  iterable = (3, 1, 4, 1)
  tuple(list(iterable)) # Noncompliant
  tuple(tuple(iterable)) # Noncompliant

def case5():
  iterable = (3, 1, 4, 1)
  list_of_list = list(iterable)
  tuple_of_list = tuple(iterable)
  set_of_list = set(iterable)
  sorted_of_list = sorted(iterable)
  set_of_set = set(iterable)
  sorted_of_sorted = sorted(iterable)
  set_of_sorted = set(iterable)

def case6():
  iterable = (3, 1, 4, 1)
  single_usage_assigned_list = list(iterable) # Noncompliant
  #                            ^^^^
  list_of_list = list(single_usage_assigned_list) 
  #              ^^^^< {{A redundant call is done here.}}
  single_usage_assigned_not_sensitive = something_else(iterable)
  list_of_list = list(single_usage_assigned_not_sensitive)
  multiple_usage_assigned_list = list(iterable)
  multiple_usage_assigned_list.append(5)
  list_of_list = list(multiple_usage_assigned_list)

def case7():
  iterable = (3, 1, 4, 1)
  sorted(reversed(iterable)) # OK, it should be raided by S7510
  reversed(sorted(iterable)) # OK, it should be raided by S7510
  list(set(iterable))
