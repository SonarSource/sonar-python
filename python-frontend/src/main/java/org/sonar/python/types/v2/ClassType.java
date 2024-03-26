package org.sonar.python.types.v2;

import java.util.List;

/**
 * ClassType
 */
public record ClassType(
  List<Member> members,
  List<PythonType> attributes,
  List<PythonType> superClasses,
  List<PythonType> typeVars) implements PythonType{
}
