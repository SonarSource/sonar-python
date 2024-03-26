package org.sonar.python.types.v2;

import java.util.List;

public record UnionType(List<PythonType> candidates) implements PythonType {
}
