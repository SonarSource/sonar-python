package org.sonar.python.types;

import org.sonar.plugins.python.api.types.InferredType;

public class PyTypeDetailedInfo {
  private final String raw;
  private InferredType inferredType;

  public PyTypeDetailedInfo(String raw) {
    this.raw = raw;
  }

  public String raw() {
    return raw;
  }

  public InferredType inferredType() {
    if (inferredType == null) {
      inferredType = PyTypeTypeGrammar.getTypeFromString(raw);
    }
    return inferredType;
  }
}
