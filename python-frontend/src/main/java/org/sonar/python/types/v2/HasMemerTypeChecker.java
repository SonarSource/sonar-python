package org.sonar.python.types.v2;

public record HasMemerTypeChecker(String memberName) implements TypeChecker {
  @Override
  public TriBool check(PythonType pythonType) {
    if (pythonType instanceof ClassType classType) {
      return classType.instancesHaveMember(memberName);
    }
    return TriBool.FALSE;
  }
}
