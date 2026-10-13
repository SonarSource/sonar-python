/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.plugins.python.api.types.v2.PythonType;

public abstract class Propagation {
  private final Set<SymbolV2> variableDependencies;
  private final Set<Propagation> dependents;

  private final SymbolV2 lhsSymbol;
  private final Name lhsName;

  protected Propagation(SymbolV2 lhsSymbol, Name lhsName) {
    this.lhsSymbol = lhsSymbol;
    this.lhsName = lhsName;
    this.variableDependencies = new HashSet<>();
    this.dependents = new HashSet<>();
  }


  /** Returns the enclosing evaluation scope of a name in an attached AST. */
  @Nonnull
  Tree scopeTree(Name name) {
    return scopeTree(name, null);
  }

  @Nonnull
  static Tree scopeTree(Name name, @Nullable Tree excludedDefinition) {
    Tree start = name;
    Tree ancestor = TreeUtils.firstAncestor(name, t -> t.is(Tree.Kind.DECORATOR, Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT, Tree.Kind.CLASSDEF));
    if (ancestor != null && ancestor.is(Tree.Kind.DECORATOR)) {
      // Decorators are evaluated in the scope enclosing the decorated definition.
      start = Objects.requireNonNull(TreeUtils.firstAncestor(ancestor, t -> t.is(Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF)),
        "A decorator must belong to a function or class definition");
    }
    return Objects.requireNonNull(TreeUtils.firstAncestor(start,
      t -> !t.equals(excludedDefinition) && t.is(Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT, Tree.Kind.CLASSDEF)),
      "A name must have an enclosing evaluation scope");
  }

  boolean areDependenciesReady(Set<SymbolV2> initializedVars) {
    return initializedVars.containsAll(variableDependencies);
  }

  Set<Propagation> dependents() {
    return dependents;
  }

  public Name lhsName() {
    return lhsName;
  }

  public abstract PythonType rhsType();

  public SymbolV2 lhsSymbol() {
    return lhsSymbol;
  }

  public void addVariableDependency(SymbolV2 dependency) {
    variableDependencies.add(dependency);
  }

  public void addDependent(Propagation dependent) {
    dependents.add(dependent);
  }

}
