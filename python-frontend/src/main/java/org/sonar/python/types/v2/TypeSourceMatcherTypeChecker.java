package org.sonar.python.types.v2;

public record TypeSourceMatcherTypeChecker(TypeSource typeSource) implements TypeChecker {
  @Override
  public TriBool check(PythonType pythonType) {
    return pythonType.typeSource() == typeSource ? TriBool.TRUE : TriBool.FALSE;
  }
}
