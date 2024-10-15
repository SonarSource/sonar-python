package org.sonar.python.types.v2;

public record UnresolvedImportType(String importPath) implements PythonType {
}
