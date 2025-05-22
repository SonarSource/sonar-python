
def case1():
  data = [3, 1, 4, 1, 5, 9]
  result = reversed(data)

def case2():
  data = [3, 1, 4, 1, 5, 9]
  result = reversed(sorted(data)) # Noncompliant

def case3():
  data = [3, 1, 4, 1, 5, 9]
  sorted_data = sorted(data)
  result = reversed(sorted_data) # Noncompliant

def case4():
  data = [3, 1, 4, 1, 5, 9]
  modified = sorted(data)
  modified.append(3)
  result = reversed(modified)

def case5():
  data = [3, 1, 4, 1, 5, 9]
  result = reversed(list(data))

def case6():
  data = [3, 1, 4, 1, 5, 9]
  list_data = list(data)
  result = reversed(list_data)
