class MyNewStyleClass(object):
	pass

class MyClassSubclassingAnotherClass(PossiblyNewStyleClass):
	pass

class MyClassWithNoSuperClass: # Noncompliant {{Add inheritance from "object" or some other new-style class.}}
#     ^^^^^^^^^^^^^^^^^^^^^^^
	pass

class MyOtherClassWithNoSuperClass(): # Noncompliant
	pass
