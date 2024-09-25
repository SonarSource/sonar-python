def correctly_named_function():
	pass

def Badly_Named_Function(): # Noncompliant {{Rename function "Badly_Named_Function" to match the regular expression ^[a-z_][a-z0-9_]*$.}}
#   ^^^^^^^^^^^^^^^^^^^^
	pass

def correct_name_with_digits1():
	pass

def long_function_name_is_still_correct():
	pass

class MyClass:
	def This_Is_A_Method():
		pass
class A():
    if 1:
        def Badly_Named(self): # compliant, this is a method, not a function
            pass
class B(SuperClass):
    if 1:
        def Badly_Named(self): # compliant, this is a method, not a function
            pass

def db(): ...  # OK

def setUpModule(): # FP
    ...
def tearDownModule(): # FP
    ...