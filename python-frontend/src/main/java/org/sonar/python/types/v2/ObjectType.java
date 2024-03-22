package org.sonar.python.types.v2;

import java.util.List;

public record ObjectType(PythonType type, List<PythonType> attributes) implements PythonType {
}
