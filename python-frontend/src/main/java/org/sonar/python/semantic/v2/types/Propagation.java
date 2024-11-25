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

import java.util.HashSet;
import java.util.Set;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.PythonType;

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


  Tree scopeTree(Name name) {
    return TreeUtils.firstAncestor(name, t ->  t.is(Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT, Tree.Kind.CLASSDEF));
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
