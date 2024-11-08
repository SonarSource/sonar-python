from uselessStatementImported import ClassWithProperty

ClassWithProperty().method # Noncompliant
ClassWithProperty().my_property # OK property access that is not a constant and might have side effects
