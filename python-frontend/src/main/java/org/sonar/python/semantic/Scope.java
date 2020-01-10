/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.semantic;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;

class Scope {

  final Tree rootTree;
  private final Scope parent;
  private final Map<String, Symbol> symbolsByName = new HashMap<>();
  private final Set<Symbol> symbols = new HashSet<>();
  final Set<Symbol> builtinSymbols = new HashSet<>();
  private final Set<String> globalNames = new HashSet<>();
  private final Set<String> nonlocalNames = new HashSet<>();
  final Map<String, SymbolImpl> instanceAttributesByName = new HashMap<>();

  Scope(@Nullable Scope parent, Tree rootTree) {
    this.parent = parent;
    this.rootTree = rootTree;
  }

  Set<Symbol> symbols() {
    return Collections.unmodifiableSet(symbols);
  }

  void createBuiltinSymbol(String name) {
    SymbolImpl symbol = new SymbolImpl(name, name);
    symbols.add(symbol);
    builtinSymbols.add(symbol);
    symbolsByName.put(name, symbol);
  }

  void createSymbolsFromWildcardImport(Set<Symbol> importedSymbols) {
    importedSymbols.forEach(symbol -> {
      symbols.add(symbol);
      symbolsByName.put(symbol.name(), symbol);
    });
  }

  void createSelfParameter(Parameter parameter) {
    Name nameTree = parameter.name();
    if (nameTree != null) {
      String symbolName = nameTree.name();
      SymbolImpl symbol = new SelfSymbolImpl(symbolName, parent);
      symbols.add(symbol);
      symbolsByName.put(symbolName, symbol);
      symbol.addUsage(nameTree, Usage.Kind.PARAMETER);
    }
  }

  void addBindingUsage(Name nameTree, Usage.Kind kind, @Nullable String fullyQualifiedName) {
    String symbolName = nameTree.name();
    if (!symbolsByName.containsKey(symbolName) && !globalNames.contains(symbolName) && !nonlocalNames.contains(symbolName)) {
      SymbolImpl symbol = new SymbolImpl(symbolName, fullyQualifiedName);
      symbols.add(symbol);
      symbolsByName.put(symbolName, symbol);
    }
    SymbolImpl symbol = resolve(symbolName);
    if (symbol != null) {
      if (fullyQualifiedName != null && !fullyQualifiedName.equals(symbol.fullyQualifiedName)) {
        symbol.fullyQualifiedName = null;
      }
      if (fullyQualifiedName == null && symbol.fullyQualifiedName != null) {
        symbol.fullyQualifiedName = null;
      }
      symbol.addUsage(nameTree, kind);
    }
  }

  @CheckForNull
  SymbolImpl resolve(String symbolName) {
    Symbol symbol = symbolsByName.get(symbolName);
    if (parent == null || symbol != null) {
      return (SymbolImpl) symbol;
    }
    return parent.resolve(symbolName);
  }

  void addGlobalName(String name) {
    globalNames.add(name);
  }

  void addNonLocalName(String name) {
    nonlocalNames.add(name);
  }
}
