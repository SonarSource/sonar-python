correct_lamda = lambda p1, p2, p3, p4, p5, p6, p7: p1

incorrect_lamda = lambda p1, p2, p3, p4, p5, p6, p7, p8: p1 # Noncompliant {{Lambda has 8 parameters, which is greater than the 7 authorized.}}
#                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def correct_function(p1, p2, p3, p4, p5, p6, p7):
	pass

def incorrect_function(p1, p2, p3, p4, p5, p6, p7, p8): # Noncompliant {{Function "incorrect_function" has 8 parameters, which is greater than the 7 authorized.}}
#                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	pass

class MyClass:
	def correct_method(p1, p2, p3, p4, p5, p6, p7):
		pass

	def incorrect_method(p1, p2, p3, p4, p5, p6, p7, p8): # Noncompliant {{Method "incorrect_method" has 8 parameters, which is greater than the 7 authorized.}}
		pass

def star_parameter(p1, p2, p3, p4, p5, p6, p7, *p8): # Noncompliant
	pass
