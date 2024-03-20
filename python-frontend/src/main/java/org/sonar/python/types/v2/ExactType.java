package org.sonar.python.types.v2;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public record ExactType(String name, List<PythonType> parameters, Map<String, Object> attributes) implements PythonType {
  @Override
  public boolean isPrimitive() {
    return Constants.PRIMITIVE_TYPE_NAMES.contains(name);
  }

  @Override
  public boolean isUnknown() {
    return Objects.isNull(name);
  }

  @Override
  public boolean isNone() {
    return Constants.NONE_TYPE_NAME.equals(name);
  }

}
