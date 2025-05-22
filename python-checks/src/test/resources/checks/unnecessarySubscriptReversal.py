
def case1():
  data = [3, 1, 4, 1, 5, 9]
  reversed(data[::-1]) # Noncompliant
  sorted(data[::-1]) # Noncompliant
  set(data[::-1]) # Noncompliant
  reversed(data[1::-1])
  sorted(data[:1:-1])
  set(data[1:1:-1])
  set(data[::])
  reversed(data)
  sorted(data)
  set(data)

def case2():
  data = [3, 1, 4, 1, 5, 9]
  reversed_data = data[::-1]
  sorted(reversed_data) # Noncompliant
  sorted(data[:3,3:])
  reversed_modified_data = data[::-1]
  reversed_modified_data.append(10)
  sorted(reversed_modified_data)
  some_data = data
  sorted(some_data)

def case3():
  data = [3, 1, 4, 1, 5, 9]
  reversed(data)[::-1] # Noncompliant
  sorted(data)[::-1] # Noncompliant
  set(data)[::-1] # Noncompliant

def case4():
  data = [3, 1, 4, 1, 5, 9]
  set(data[::+1])
  set(data[::-2])
  set(data[::1])
  set(data[::-b])
  list(data[::-1])

def case5():
  data = [3, 1, 4, 1, 5, 9]
  reversed()
  sorted(data)[1]
  sorted(reverse=True)
  set()
