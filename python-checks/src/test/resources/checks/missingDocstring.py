"""
This is a module docstring
"""

def function_with_docstring():
	"""Function docstring"""
	pass

def function_without_docstring():
	pass

def function_with_empty_docstring():
	""" """
	pass

def function_without_docstring_and_compound_statement():
	if a: pass

def one_line_function_with_docstring(): "this is a docstring"; pass

def one_line_function_without_docstring(): pass; "not a docstring"

class ClassWithDocstring:
	"""This is a docstring"""
	pass

class ClassWithoutDocstring:
	def method_with_docstring():
		""" doc """
		pass

	def method_without_docstring():
		pass
