"""
This is a module docstring
"""

def function_with_docstring():
	"""This is a function docstring"""
	pass

def function_without_docstring():
	pass

def function_with_empty_docstring():
	""" """
	pass

def one_line_function_with_docstring(): "This is a function docstring"; pass

def one_line_function_without_docstring(): pass; "This is not a docstring"

class ClassWithDocstring:
	"""This is a class docstring"""
	pass

class ClassWithoutDocstring:
	def method_with_docstring():
		''' This is a method docstring '''
		pass

	def method_without_docstring():
		pass

def function_with_unusual_docstring(): "This is still" \
							 "a docstring"; pass

def function_with_no_docstring(): "This is not", "a docstring"; pass
