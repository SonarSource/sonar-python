package org.sonar.python.types.v2;

import java.util.List;

public class UnionType implements PythonType {
    List<PythonType> candidates;
}
