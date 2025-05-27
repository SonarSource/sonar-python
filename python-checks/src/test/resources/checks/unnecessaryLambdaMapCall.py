
def case1():
  nums = [1, 2, 3, 4]
  gen_map = map(lambda x: x + 1, nums) # Noncompliant
  l = lambda x: x + 1
  gen_map = map(l, nums) # Noncompliant


def case2():
  nums = [1, 2, 3, 4]
  gen_map = not_a_map_call(lambda x: x + 1, nums)

def case3():
  nums = [1, 2, 3, 4]
  l = not_a_lambda
  gen_map = map(l, nums)

def case4():
  gen_map = map()

def case5():
  nums = [1, 2, 3, 4]
  l = lambda x: x + 1
  usage_of_lambda(l)
  gen_map = map(l, nums)

def case5():
  nums = [1, 2, 3, 4]
  l = "not a lambda with more usages"
  x = l
  gen_map = map(l, nums)