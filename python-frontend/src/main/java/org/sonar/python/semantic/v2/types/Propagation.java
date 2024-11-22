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
import java.util.Set;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.PythonType;

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


  Tree scopeTree(Name name) {
    return TreeUtils.firstAncestor(name, t ->  t.is(Tree.Kind.FUNCDEF, Tree.Kind.FILE_INPUT, Tree.Kind.CLASSDEF));
  }

  boolean areDependenciesReady(Set<SymbolV2> initializedVars) {
    return initializedVars.containsAll(variableDependencies);
  }

  Set<Propagation> dependents() {
    return dependents;
  }

  public abstract Name lhsName();

  public abstract PythonType rhsType();

  public SymbolV2 lhsSymbol() {
    return lhsSymbol;
  }
}
