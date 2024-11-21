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
package org.sonar.python.semantic.v2.types;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.Set;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.types.HasTypeDependencies;
import org.sonar.python.types.v2.PythonType;

public class Assignment extends Propagation {

  Expression rhs;
  Map<SymbolV2, Set<Propagation>> propagationsByLhs;

  public Assignment(SymbolV2 lhsSymbol, Name lhsName, Expression rhs, Map<SymbolV2, Set<Propagation>> propagationsByLhs) {
    super(lhsSymbol, lhsName);
    this.rhs = rhs;
    this.propagationsByLhs = propagationsByLhs;
  }

  void computeDependencies(Set<SymbolV2> trackedVars) {
    Deque<Expression> workList = new ArrayDeque<>();
    workList.push(rhs);
    while (!workList.isEmpty()) {
      Expression e = workList.pop();
      if (e instanceof Name name) {
        SymbolV2 symbol = name.symbolV2();
        if (symbol != null && trackedVars.contains(symbol)) {
          variableDependencies.add(symbol);
          propagationsByLhs.get(symbol).forEach(a -> a.dependents.add(this));
        }
      } else if (e instanceof HasTypeDependencies hasTypeDependencies) {
        workList.addAll(hasTypeDependencies.typeDependencies());
      }
    }
  }

  public Name lhsName() {
    return lhsName;
  }

  @Override
  public PythonType rhsType() {
    return rhs.typeV2();
  }

  public Expression rhs() {
    return rhs;
  }
}
