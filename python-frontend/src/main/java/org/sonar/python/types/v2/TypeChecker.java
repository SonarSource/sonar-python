package org.sonar.python.types.v2;

public interface TypeChecker {

  TriBool check(PythonType pythonType);

}
