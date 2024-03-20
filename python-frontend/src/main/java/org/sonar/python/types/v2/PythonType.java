package org.sonar.python.types.v2;

/**
 * PythonType
 */
public interface PythonType {
  boolean isPrimitive();

  boolean isUnknown();

  boolean isNone();
}
