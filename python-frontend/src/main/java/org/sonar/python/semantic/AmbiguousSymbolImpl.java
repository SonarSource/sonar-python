/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Tree;

public class AmbiguousSymbolImpl extends SymbolImpl implements AmbiguousSymbol {

  private final Set<Symbol> symbols;
  private Map<Tree, Symbol> symbolsByDeclarationTree = new HashMap<>();

  private AmbiguousSymbolImpl(String name, @Nullable String fullyQualifiedName, Set<Symbol> symbols) {
    super(name, fullyQualifiedName);
    setKind(Kind.AMBIGUOUS);
    this.symbols = symbols;
  }

  public static AmbiguousSymbol create(Set<Symbol> symbols) {
    if (symbols.size() < 2) {
      throw new IllegalArgumentException("Ambiguous symbol should contain at least two symbols");
    }
    Symbol firstSymbol = symbols.iterator().next();
    if (!symbols.stream().map(Symbol::name).allMatch(symbolName -> symbolName.equals(firstSymbol.name()))) {
      throw new IllegalArgumentException("Ambiguous symbol should contain symbols with the same name");
    }
    if (!symbols.stream().map(Symbol::fullyQualifiedName).allMatch(fqn -> Objects.equals(firstSymbol.fullyQualifiedName(), fqn))) {
      return new AmbiguousSymbolImpl(firstSymbol.name(), null, symbols);
    }
    return new AmbiguousSymbolImpl(firstSymbol.name(), firstSymbol.fullyQualifiedName(), symbols);
  }

  @Override
  public Set<Symbol> symbols() {
    return symbols;
  }

  void setSymbolsByDeclarationTree(Map<Tree, Symbol> symbolsByDeclarationTree) {
    this.symbolsByDeclarationTree = symbolsByDeclarationTree;
  }

  public Symbol getSymbol(Tree tree) {
    return symbolsByDeclarationTree.get(tree);
  }

  @Override
  AmbiguousSymbolImpl copyWithoutUsages() {
    Set<SymbolImpl> copiedAlternativeSymbols = symbols.stream()
      .map(SymbolImpl.class::cast)
      .map(SymbolImpl::copyWithoutUsages)
      .collect(Collectors.toSet());
    return ((AmbiguousSymbolImpl) create(Collections.unmodifiableSet(copiedAlternativeSymbols)));
  }
}
