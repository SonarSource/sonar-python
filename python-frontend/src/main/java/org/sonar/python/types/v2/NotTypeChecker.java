package org.sonar.python.types.v2;

public record NotTypeChecker(TypeChecker nested) implements TypeChecker{
  @Override
  public TriBool check(PythonType pythonType) {
    return nested.check(pythonType).not();
  }
}
