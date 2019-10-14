# simple check for custom rules

class A:
  def fun(): # NOK - function definition
    pass

  for foo in bar: # NOK - for statement
    pass
  else:
    pass
