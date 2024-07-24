package org.sonar.python.types.v2;

import java.util.List;

public record AndTypeChecker(List<TypeChecker> checkers) implements TypeChecker {
  @Override
  public TriBool check(PythonType pythonType) {
    return checkers.stream().map(checker -> checker.check(pythonType)).allMatch(TriBool.TRUE::equals) ? TriBool.TRUE : TriBool.FALSE;
  }
}
