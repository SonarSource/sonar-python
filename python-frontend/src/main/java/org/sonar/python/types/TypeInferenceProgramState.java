/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.types;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.cfg.fixpoint.ProgramState;

public class TypeInferenceProgramState implements ProgramState {
  // Using Set of types instead of "Union type" in order to represent BOTTOM as an empty set
  private final Map<Symbol, Set<InferredType>> inferredTypesBySymbol;

  TypeInferenceProgramState() {
    this.inferredTypesBySymbol = new HashMap<>();
  }

  void setTypes(Symbol symbol, Set<InferredType> types) {
    inferredTypesBySymbol.put(symbol, types);
  }

  Set<InferredType> getTypes(@Nullable Symbol symbol) {
    return inferredTypesBySymbol.getOrDefault(symbol, Collections.emptySet());
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    TypeInferenceProgramState that = (TypeInferenceProgramState) o;
    return Objects.equals(inferredTypesBySymbol, that.inferredTypesBySymbol);
  }

  @Override
  public int hashCode() {
    return Objects.hash(inferredTypesBySymbol);
  }

  @Override
  public String toString() {
    StringBuilder result = new StringBuilder();
    for (Map.Entry<Symbol, Set<InferredType>> entry : inferredTypesBySymbol.entrySet()) {
      Symbol symbol = entry.getKey();
      Set<InferredType> inferredTypes = entry.getValue();
      result
        .append(symbol.name()).append(" = ")
        .append(inferredTypes.stream().map(Objects::toString).collect(Collectors.joining(", ")))
        .append(System.lineSeparator());
    }
    return result.toString();
  }

  @Override
  public ProgramState join(ProgramState otherState) {
    TypeInferenceProgramState result = new TypeInferenceProgramState();
    ((TypeInferenceProgramState) otherState).inferredTypesBySymbol.forEach(((symbol, types) -> {
      HashSet<InferredType> union = new HashSet<>(types);
      union.addAll(inferredTypesBySymbol.getOrDefault(symbol, Collections.emptySet()));
      result.setTypes(symbol, union);
    }));
    inferredTypesBySymbol.forEach(((symbol, types) -> {
      if (!result.inferredTypesBySymbol.containsKey(symbol)) {
        result.setTypes(symbol, types);
      }
    }));
    return result;
  }

  @Override
  public ProgramState copy() {
    return join(new TypeInferenceProgramState());
  }
}
