"""
This is a module docstring
"""

def function_with_docstring():
	"""Function docstring"""
	pass

def function_without_docstring(): # Noncompliant {{Add a docstring to this function.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^
	pass

def function_with_empty_docstring(): # Noncompliant {{The docstring for this function should not be empty.}}
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	""" """
	pass

def function_without_docstring_and_compound_statement(): # Noncompliant {{Add a docstring to this function.}}
	if a: pass

def one_line_function_with_docstring(): "this is a docstring"; pass

def one_line_function_without_docstring(): pass; "not a docstring" # Noncompliant

class ClassWithDocstring:
	"""This is a docstring"""
	pass

class ClassWithoutDocstring: # Noncompliant {{Add a docstring to this class.}}
#     ^^^^^^^^^^^^^^^^^^^^^
	def method_with_docstring():
		""" doc """
		pass

	def method_without_docstring(): # Noncompliant {{The docstring for this method should not be empty.}}
		''''''
		pass

	def method_without_docstring(): # FN Noncompliant {{Add a docstring to this method.}}
		pass
