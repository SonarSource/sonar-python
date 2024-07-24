package org.sonar.python.types.v2;

import org.sonar.python.semantic.v2.ProjectLevelTypeTable;

public record IsInstanceTypeChecker(ProjectLevelTypeTable typeTable, String fullyQualifiedName) implements TypeChecker {
  @Override
  public TriBool check(PythonType pythonType) {
    var expectedType = typeTable.getType(fullyQualifiedName);
    return pythonType.equals(expectedType) ? TriBool.TRUE : TriBool.FALSE;
  }
}
