package org.sonar.python.types.v2;

import java.util.List;

/**
 * FunctionType
 */
public record FunctionType(
  List<Member> members,
  List<PythonType> attributes,
  List<PythonType> superClasses,
  List<PythonType> typeVars,
  List<Member> parameters,
  PythonType returnType) implements PythonType {

}
