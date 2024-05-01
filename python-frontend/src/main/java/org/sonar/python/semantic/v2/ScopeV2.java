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
package org.sonar.python.semantic.v2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;

public class ScopeV2 {
  private final ScopeV2 parent;
  private final Tree rootTree;
  private final List<ScopeV2> childrenScopes;
  private final Map<String, SymbolV2> symbolsByName = new HashMap<>();
  private final Set<SymbolV2> symbols = new HashSet<>();

  public ScopeV2(@Nullable ScopeV2 parent, Tree rootTree) {
    this.parent = parent;
    this.rootTree = rootTree;
    this.childrenScopes = new ArrayList<>();
  }

  public ScopeV2 parent() {
    return parent;
  }

  public Tree root() {
    return rootTree;
  }

  void addBindingUsage(Name nameTree, UsageV2.Kind kind, @Nullable String fullyQualifiedName) {
    String symbolName = nameTree.name();
    if (!isExistingSymbol(symbolName)) {
      SymbolV2 symbol = new SymbolV2(symbolName);
      symbols.add(symbol);
      symbolsByName.put(symbolName, symbol);
    }
    SymbolV2 symbol = resolve(symbolName);
    if (symbol != null) {
      symbol.addUsage(nameTree, kind);
    }
  }

  @CheckForNull
  SymbolV2 resolve(String symbolName) {
    SymbolV2 symbol = symbolsByName.get(symbolName);
    if (parent == null || symbol != null) {
      return symbol;
    }
    // parent scope of a symbol inside of a class its the classes parent scope
    if (parent.rootTree.is(Tree.Kind.CLASSDEF)) {
      return parent.parent.resolve(symbolName);
    }
    return parent.resolve(symbolName);
  }

  private boolean isExistingSymbol(String symbolName) {
    //return symbolsByName.containsKey(symbolName) || globalNames.contains(symbolName) || nonlocalNames.contains(symbolName);
    return symbolsByName.containsKey(symbolName);
  }

  void createSelfParameter(Parameter parameter) {
    Name nameTree = parameter.name();
    if (nameTree == null) {
      return;
    }
    String symbolName = nameTree.name();
    //TODO: Check with SelfSymbolImpl
    SymbolV2 symbol = new SymbolV2(symbolName);
    symbols.add(symbol);
    symbolsByName.put(symbolName, symbol);
    symbol.addUsage(nameTree, UsageV2.Kind.PARAMETER);
  }

  public Map<String, SymbolV2> symbols() {
    return symbolsByName;
  }

  public List<ScopeV2> childrenScopes() {
    return childrenScopes;
  }
}
