
def case1():
  numbers = [1, 5, 0, 10]
  all([x > 2 for x in numbers])  # Noncompliant
  any([x > 2 for x in numbers])  # Noncompliant

def case2():
  numbers = [1, 5, 0, 10]
  d1 = [x > 2 for x in numbers]
  all(d1) # Noncompliant
  d2 = [x > 2 for x in numbers]
  d2.append(3)
  any(d2)

def case3():
  numbers = [1, 5, 0, 10]
  all(x > 2 for x in numbers)
  any(x > 2 for x in numbers)

def case4():
  numbers = [1, 5, 0, 10]
  d1 = {x > 2 for x in numbers}
  all(d1)
  d2 = {x > 2 for x in numbers}
  any(d2)

def case5():
  numbers = [1, 5, 0, 10]
  all({x > 2 for x in numbers})
  any({x > 2 for x in numbers})