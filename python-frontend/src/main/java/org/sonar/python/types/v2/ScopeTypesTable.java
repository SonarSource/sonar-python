package org.sonar.python.types.v2;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

public class ScopeTypesTable {
  private final ScopeTypesTable parent;
  private final List<PythonType> types;

  public ScopeTypesTable(ScopeTypesTable parent) {
    this.parent = parent;
    this.types = new ArrayList<>();
  }

  public Optional<PythonType> findType(Predicate<PythonType> predicate) {
    return types.stream()
      .filter(predicate)
      .findFirst()
      .or(() -> Optional.ofNullable(parent)
        .flatMap(p -> p.findType(predicate))
      ).or(Optional::empty);
  }

  public void registerType(PythonType type) {
    types.add(type);
  }
}
