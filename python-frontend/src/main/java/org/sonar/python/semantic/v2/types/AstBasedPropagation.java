/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.semantic.v2.types;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.TypeTable;
import org.sonar.python.types.v2.PythonType;

public class AstBasedPropagation {
  private final Map<SymbolV2, Set<Propagation>> propagationsByLhs;
  private final TypeTable typeTable;

  public AstBasedPropagation(Map<SymbolV2, Set<Propagation>> propagationsByLhs, TypeTable typeTable) {
    this.propagationsByLhs = propagationsByLhs;
    this.typeTable = typeTable;
  }

  public Map<SymbolV2, Set<PythonType>> processPropagations(Set<SymbolV2> trackedVars) {
    Set<Propagation> propagations = new HashSet<>();
    Set<SymbolV2> initializedVars = new HashSet<>();

    propagationsByLhs.forEach((lhs, props) -> {
      if (trackedVars.contains(lhs)) {
        props.stream()
          .filter(Assignment.class::isInstance)
          .map(Assignment.class::cast)
          .forEach(a -> a.computeDependencies(trackedVars));
        propagations.addAll(props);
      }
    });

    applyPropagations(propagations, initializedVars, true);
    applyPropagations(propagations, initializedVars, false);
    return propagations.stream().collect(Collectors.groupingBy(Propagation::lhsSymbol, Collectors.mapping(Propagation::rhsType, Collectors.toSet())));
  }

  private static void applyPropagations(Set<Propagation> propagations, Set<SymbolV2> initializedVars, boolean checkDependenciesReadiness) {
    Set<Propagation> workSet = new HashSet<>(propagations);
    while (!workSet.isEmpty()) {
      Iterator<Propagation> iterator = workSet.iterator();
      Propagation propagation = iterator.next();
      iterator.remove();
      if (!checkDependenciesReadiness || propagation.areDependenciesReady(initializedVars)) {
        boolean learnt = propagation.propagate(initializedVars);
        if (learnt) {
          workSet.addAll(propagation.dependents());
        }
      }
    }
  }
}
