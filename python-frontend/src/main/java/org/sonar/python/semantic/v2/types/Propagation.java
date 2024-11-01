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
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.SymbolV2Utils;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

public abstract class Propagation {

  final Set<SymbolV2> variableDependencies;
  final Set<Propagation> dependents;

  final SymbolV2 lhsSymbol;
  final Name lhsName;

  protected Propagation(SymbolV2 lhsSymbol, Name lhsName) {
    this.lhsSymbol = lhsSymbol;
    this.lhsName = lhsName;
    this.variableDependencies = new HashSet<>();
    this.dependents = new HashSet<>();
  }

  /**
   * This is used for AST-based type inference in try/catch statements
   * @return true if the propagation effectively changed the inferred type of assignment LHS
   */
  boolean propagate(Set<SymbolV2> initializedVars) {
    PythonType rhsType = rhsType();
    Tree scopeTree = scopeTree(lhsName);
    if (initializedVars.add(lhsSymbol)) {
      getSymbolNonDeclarationUsageTrees(lhsSymbol)
        .filter(NameImpl.class::isInstance)
        .map(NameImpl.class::cast)
        // Avoid propagation to usages in nested scopes, as this may lead to FPs
        .filter(n -> isInSameScope(n, scopeTree))
        .forEach(n -> n.typeV2(rhsType));
      return true;
    } else {
      PythonType currentType = currentType(lhsName);
      if (currentType == null) {
        return false;
      }
      PythonType newType = UnionType.or(rhsType, currentType);
      getSymbolNonDeclarationUsageTrees(lhsSymbol)
        .filter(NameImpl.class::isInstance)
        .map(NameImpl.class::cast)
        .filter(n -> isInSameScope(n, scopeTree))
        .forEach(n -> n.typeV2(newType));
      return !newType.equals(currentType);
    }
  }

  private boolean isInSameScope(Name n, Tree scopeTree) {
    return Optional.ofNullable(scopeTree(n)).filter(scopeTree::equals).isPresent();
  }

  Tree scopeTree(Name name) {
    return TreeUtils.firstAncestor(name, t ->  t.is(Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT, Tree.Kind.CLASSDEF));
  }

  public static Stream<Tree> getSymbolNonDeclarationUsageTrees(SymbolV2 symbol) {
    return symbol.usages()
      .stream()
      // Function and class definition names will always have FunctionType and ClassType respectively
      // so they are filtered out of type propagation
      .filter(u -> !SymbolV2Utils.isDeclaration(u))
      .map(UsageV2::tree);
  }

  boolean areDependenciesReady(Set<SymbolV2> initializedVars) {
    return initializedVars.containsAll(variableDependencies);
  }

  Set<Propagation> dependents() {
    return dependents;
  }

  public abstract Name lhsName();

  public abstract PythonType rhsType();

  @CheckForNull
  static PythonType currentType(Name lhsName) {
    return Optional.ofNullable(lhsName.symbolV2())
      .stream()
      .map(SymbolV2::usages)
      .flatMap(List::stream)
      .filter(u -> !SymbolV2Utils.isDeclaration(u))
      .map(UsageV2::tree)
      .filter(Expression.class::isInstance)
      .map(Expression.class::cast)
      .findFirst()
      .map(Expression::typeV2)
      .orElse(null);
  }

  public SymbolV2 lhsSymbol() {
    return lhsSymbol;
  }
}
