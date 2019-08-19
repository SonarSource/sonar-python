def correctly_named_function():
	pass

def Badly_Named_Function(): # Noncompliant {{Rename function "Badly_Named_Function" to match the regular expression ^[a-z_][a-z0-9_]{2,}$.}}
#   ^^^^^^^^^^^^^^^^^^^^
	pass

def correct_name_with_digits1():
	pass

def long_function_name_is_still_correct():
	pass

class MyClass:
	def This_Is_A_Method():
		pass
