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

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.SymbolV2Utils;
import org.sonar.python.semantic.v2.TypeTable;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.HasTypeDependencies;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

public class AstBasedTypeInference {
  private final Map<SymbolV2, Set<Propagation>> propagationsByLhs;
  private final Propagator propagator;

  public AstBasedTypeInference(Map<SymbolV2, Set<Propagation>> propagationsByLhs, TypeTable typeTable) {
    this.propagationsByLhs = propagationsByLhs;
    this.propagator = new Propagator(typeTable);
  }

  public Map<SymbolV2, Set<PythonType>> process(Set<SymbolV2> trackedVars) {
    computePropagationDependencies(trackedVars);

    Set<SymbolV2> initializedVars = new HashSet<>();
    Set<Propagation> propagations = getTrackedPropagation(trackedVars);

    applyPropagations(propagations, initializedVars, true);
    applyPropagations(propagations, initializedVars, false);
    return propagations.stream().collect(Collectors.groupingBy(Propagation::lhsSymbol, Collectors.mapping(Propagation::rhsType, Collectors.toSet())));
  }

  private void computePropagationDependencies(Set<SymbolV2> trackedVars) {
    propagationsByLhs.forEach((lhs, props) -> {
      if (trackedVars.contains(lhs)) {
        props.stream()
          .filter(Assignment.class::isInstance)
          .map(Assignment.class::cast)
          .forEach(assignment -> computeDependencies(assignment, trackedVars));
      }
    });
  }

  private void computeDependencies(Assignment assignment, Set<SymbolV2> trackedVars) {
    Deque<Expression> workList = new ArrayDeque<>();
    workList.push(assignment.rhs());
    while (!workList.isEmpty()) {
      Expression e = workList.pop();
      if (e instanceof Name name) {
        SymbolV2 symbol = name.symbolV2();
        if (symbol != null && trackedVars.contains(symbol)) {
          assignment.addVariableDependency(symbol);
          propagationsByLhs.get(symbol).forEach(propagation -> propagation.addDependent(assignment));
        }
      } else if (e instanceof HasTypeDependencies hasTypeDependencies) {
        workList.addAll(hasTypeDependencies.typeDependencies());
      }
    }
  }

  private Set<Propagation> getTrackedPropagation(Set<SymbolV2> trackedVars) {
    Set<Propagation> trackedPropagations = new HashSet<>();
    propagationsByLhs.forEach((lhs, propagations) -> {
      if (trackedVars.contains(lhs)) {
        trackedPropagations.addAll(propagations);
      }
    });
    return trackedPropagations;
  }

  private void applyPropagations(Set<Propagation> propagations, Set<SymbolV2> initializedVars, boolean checkDependenciesReadiness) {
    Set<Propagation> workSet = new HashSet<>(propagations);
    while (!workSet.isEmpty()) {
      Iterator<Propagation> iterator = workSet.iterator();
      Propagation propagation = iterator.next();
      iterator.remove();
      if (!checkDependenciesReadiness || propagation.areDependenciesReady(initializedVars)) {
        boolean learnt = propagator.propagate(propagation, initializedVars);
        if (learnt) {
          workSet.addAll(propagation.dependents());
        }
      }
    }
  }


  private record Propagator(TypeTable typeTable) {

    /**
     * @return true if the propagation effectively changed the inferred type of assignment LHS
     */
    public boolean propagate(Propagation propagation, Set<SymbolV2> initializedVars) {
      PythonType rhsType = propagation.rhsType();
      Name lhsName = propagation.lhsName();
      SymbolV2 lhsSymbol = propagation.lhsSymbol();
      if (initializedVars.add(lhsSymbol)) {
        propagateTypeToUsages(propagation, rhsType);
        return true;
      } else {
        PythonType currentType = currentType(lhsName);
        if (currentType == null) {
          return false;
        }
        PythonType newType = UnionType.or(rhsType, currentType);
        propagateTypeToUsages(propagation, newType);
        return !newType.equals(currentType);
      }
    }

    private void propagateTypeToUsages(Propagation propagation, PythonType newType) {
      Tree scopeTree = propagation.scopeTree(propagation.lhsName());
      getSymbolNonDeclarationUsageTrees(propagation.lhsSymbol())
        .filter(NameImpl.class::isInstance)
        .map(NameImpl.class::cast)
        // Avoid propagation to usages in nested scopes, as this may lead to FPs
        .filter(n -> isInSameScope(propagation, n, scopeTree))
        .forEach(n -> n.typeV2(newType));

      updateTree(propagation);
    }

    private void updateTree(Propagation propagation) {
      Tree scopeTree = propagation.scopeTree(propagation.lhsName());
      scopeTree.accept(new TrivialTypePropagationVisitor(typeTable));
    }

    @CheckForNull
    private static PythonType currentType(Name lhsName) {
      return Optional.ofNullable(lhsName.symbolV2())
        .stream()
        .flatMap(Propagator::getSymbolNonDeclarationUsageTrees)
        .flatMap(TreeUtils.toStreamInstanceOfMapper(Expression.class))
        .findFirst()
        .map(Expression::typeV2)
        .orElse(null);
    }

    private static Stream<Tree> getSymbolNonDeclarationUsageTrees(SymbolV2 symbol) {
      return symbol.usages()
        .stream()
        // Function and class definition names will always have FunctionType and ClassType respectively
        // so they are filtered out of type propagation
        .filter(u -> !SymbolV2Utils.isDeclaration(u))
        .map(UsageV2::tree);
    }

    private boolean isInSameScope(Propagation propagation, Name n, Tree scopeTree) {
      return Optional.ofNullable(propagation.scopeTree(n)).filter(scopeTree::equals).isPresent();
    }
  }

}
