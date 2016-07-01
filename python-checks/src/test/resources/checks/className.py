class MyClassWithCompliantName1:
	pass

class MyClass_WithNotCompliantName1: # Noncompliant {{Rename class "MyClass_WithNotCompliantName1" to match the regular expression ^[A-Z_][a-zA-Z0-9]+$.}}
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	pass

class myClassWithNotCompliantName2: # Noncompliant
	pass
