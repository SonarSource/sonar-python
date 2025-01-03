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
  private final Set<String> globalNames = new HashSet<>();
  private final Set<String> nonlocalNames = new HashSet<>();

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

  void addBindingUsage(Name nameTree, UsageV2.Kind kind) {
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
    return symbolsByName.containsKey(symbolName) || globalNames.contains(symbolName) || nonlocalNames.contains(symbolName);
  }

  void createSelfParameter(Parameter parameter) {
    Name nameTree = parameter.name();
    if (nameTree == null) {
      return;
    }
    String symbolName = nameTree.name();
    //TODO: SONARPY-1865 Represent "self"
    SymbolV2 symbol = new SymbolV2(symbolName);
    symbols.add(symbol);
    symbolsByName.put(symbolName, symbol);
    symbol.addUsage(nameTree, UsageV2.Kind.PARAMETER);
  }

  public void addGlobalName(Name name) {
    this.globalNames.add(name.name());
  }

  public void addNonLocalName(Name name) {
    this.nonlocalNames.add(name.name());
  }

  public Map<String, SymbolV2> symbols() {
    return symbolsByName;
  }

  public List<ScopeV2> childrenScopes() {
    return childrenScopes;
  }
}
