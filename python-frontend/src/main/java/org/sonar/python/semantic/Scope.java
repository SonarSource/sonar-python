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
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
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

  void addFunctionSymbol(FunctionDef functionDef, @Nullable String fullyQualifiedName) {
    String symbolName = functionDef.name().name();
    if (isExistingSymbol(symbolName)) {
      addBindingUsage(functionDef.name(), Usage.Kind.FUNC_DECLARATION, fullyQualifiedName);
    } else {
      FunctionSymbolImpl functionSymbol = new FunctionSymbolImpl(functionDef, fullyQualifiedName);
      symbols.add(functionSymbol);
      symbolsByName.put(symbolName, functionSymbol);
      functionSymbol.addUsage(functionDef.name(), Usage.Kind.FUNC_DECLARATION);
    }
  }

  private static Symbol copySymbol(String symbolName, Symbol symbol) {
    return symbol.kind() == Symbol.Kind.FUNCTION
      ? new FunctionSymbolImpl(symbolName, (FunctionSymbol) symbol)
      : new SymbolImpl(symbolName, symbol.fullyQualifiedName());
  }

  void addModuleSymbol(Name nameTree, @CheckForNull String fullyQualifiedName, Map<String, Set<Symbol>> globalSymbolsByModuleName) {
    String symbolName = nameTree.name();
    Set<Symbol> moduleExportedSymbols = globalSymbolsByModuleName.get(fullyQualifiedName);
    if (moduleExportedSymbols != null && !isExistingSymbol(symbolName)) {
      SymbolImpl moduleSymbol = new SymbolImpl(symbolName, fullyQualifiedName);
      moduleExportedSymbols.forEach(symbol -> moduleSymbol.addChildSymbol(copySymbol(symbol.name(), symbol)));
      this.symbols.add(moduleSymbol);
      symbolsByName.put(symbolName, moduleSymbol);
    }
    addBindingUsage(nameTree, Usage.Kind.IMPORT, fullyQualifiedName);
  }

  void addImportedSymbol(Name nameTree, @CheckForNull String fullyQualifiedName, Map<String, Symbol> globalSymbolsByFQN) {
    String symbolName = nameTree.name();
    Symbol globalSymbol = globalSymbolsByFQN.get(fullyQualifiedName);
    if (globalSymbol == null || isExistingSymbol(symbolName)) {
      addBindingUsage(nameTree, Usage.Kind.IMPORT, fullyQualifiedName);
    } else {
      Symbol symbol = copySymbol(symbolName, globalSymbol);
      this.symbols.add(symbol);
      symbolsByName.put(symbolName, symbol);
      ((SymbolImpl) symbol).addUsage(nameTree, Usage.Kind.IMPORT);
    }
  }

  private boolean isExistingSymbol(String symbolName) {
    return symbolsByName.containsKey(symbolName) || globalNames.contains(symbolName) || nonlocalNames.contains(symbolName);
  }

  void addBindingUsage(Name nameTree, Usage.Kind kind, @Nullable String fullyQualifiedName) {
    String symbolName = nameTree.name();
    if (!isExistingSymbol(symbolName)) {
      SymbolImpl symbol = new SymbolImpl(symbolName, fullyQualifiedName);
      symbols.add(symbol);
      symbolsByName.put(symbolName, symbol);
    }
    SymbolImpl symbol = resolve(symbolName);
    if (symbol != null) {
      if (!Symbol.Kind.OTHER.equals(symbol.kind())) {
        symbol.setKind(Symbol.Kind.OTHER);
      }
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
    if (parent.rootTree.is(Tree.Kind.CLASSDEF)) {
      return parent.parent.resolve(symbolName);
    }
    return parent.resolve(symbolName);
  }

  void addGlobalName(String name) {
    globalNames.add(name);
  }

  void addNonLocalName(String name) {
    nonlocalNames.add(name);
  }

  void addClassSymbol(ClassDef classDef, @Nullable String fullyQualifiedName) {
    String symbolName = classDef.name().name();
    if (isExistingSymbol(symbolName)) {
      addBindingUsage(classDef.name(), Usage.Kind.CLASS_DECLARATION, fullyQualifiedName);
    } else {
      ClassSymbolImpl classSymbol = new ClassSymbolImpl(symbolName, fullyQualifiedName);
      symbols.add(classSymbol);
      symbolsByName.put(symbolName, classSymbol);
      classSymbol.addUsage(classDef.name(), Usage.Kind.CLASS_DECLARATION);
    }
  }
}
