/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.semantic.v2.types;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.python.cfg.fixpoint.ProgramState;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.types.v2.PythonType;

public class TypeInferenceProgramState implements ProgramState {
  // Using Set of types instead of "Union type" in order to represent BOTTOM as an empty set
  private final Map<SymbolV2, Set<PythonType>> typesBySymbol;

  public TypeInferenceProgramState() {
    this.typesBySymbol = new HashMap<>();
  }

  public void setTypes(SymbolV2 symbol, Set<PythonType> types) {
    typesBySymbol.put(symbol, types);
  }

  Set<PythonType> getTypes(@Nullable SymbolV2 symbol) {
    return typesBySymbol.getOrDefault(symbol, Collections.emptySet());
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
    return Objects.equals(typesBySymbol, that.typesBySymbol);
  }

  @Override
  public int hashCode() {
    return Objects.hash(typesBySymbol);
  }

  @Override
  public ProgramState join(ProgramState otherState) {
    TypeInferenceProgramState result = new TypeInferenceProgramState();
    ((TypeInferenceProgramState) otherState).typesBySymbol.forEach(((symbol, types) -> {
      HashSet<PythonType> union = new HashSet<>(types);
      union.addAll(typesBySymbol.getOrDefault(symbol, Collections.emptySet()));
      result.setTypes(symbol, union);
    }));
    typesBySymbol.forEach(((symbol, types) -> {
      if (!result.typesBySymbol.containsKey(symbol)) {
        result.setTypes(symbol, types);
      }
    }));
    return result;
  }

  @Override
  public ProgramState copy() {
    return join(new TypeInferenceProgramState());
  }

  public Map<SymbolV2, Set<PythonType>> typesBySymbol() {
    return typesBySymbol;
  }
}
