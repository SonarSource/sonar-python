/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;

import static org.sonar.python.semantic.SymbolUtils.flattenAmbiguousSymbols;

public class AmbiguousSymbolImpl extends SymbolImpl implements AmbiguousSymbol {

  private final Set<Symbol> symbols;

  public AmbiguousSymbolImpl(String name, @Nullable String fullyQualifiedName, Set<Symbol> symbols) {
    super(name, fullyQualifiedName);
    setKind(Kind.AMBIGUOUS);
    this.symbols = symbols;
  }

  public static AmbiguousSymbol create(Set<Symbol> symbols) {
    if (symbols.size() < 2) {
      throw new IllegalArgumentException("Ambiguous symbol should contain at least two symbols");
    }
    Symbol firstSymbol = symbols.iterator().next();
    String resultingSymbolName = firstSymbol.name();
    if (!symbols.stream().map(Symbol::name).allMatch(symbolName -> symbolName.equals(firstSymbol.name()))) {
      if (!symbols.stream().map(Symbol::fullyQualifiedName).allMatch(fqn -> Objects.equals(firstSymbol.fullyQualifiedName(), fqn))) {
        throw new IllegalArgumentException("Ambiguous symbol should contain symbols with the same name");
      }
      // Here we have symbols having same FQN but different local names, so we cannot assign any name to resulting value
      resultingSymbolName = "";
    }
    if (!symbols.stream().map(Symbol::fullyQualifiedName).allMatch(fqn -> Objects.equals(firstSymbol.fullyQualifiedName(), fqn))) {
      return new AmbiguousSymbolImpl(resultingSymbolName, null, symbols);
    }
    return new AmbiguousSymbolImpl(resultingSymbolName, firstSymbol.fullyQualifiedName(), flattenAmbiguousSymbols(symbols));
  }

  public static AmbiguousSymbol create(Symbol... symbols) {
    return create(new HashSet<>(Arrays.asList(symbols)));
  }

  @Override
  public Set<Symbol> alternatives() {
    return symbols;
  }

  @Override
  public AmbiguousSymbolImpl copyWithoutUsages() {
    Set<SymbolImpl> copiedAlternativeSymbols = symbols.stream()
      .map(SymbolImpl.class::cast)
      .map(SymbolImpl::copyWithoutUsages)
      .collect(Collectors.toSet());
    return ((AmbiguousSymbolImpl) create(Collections.unmodifiableSet(copiedAlternativeSymbols)));
  }

  @Override
  public void removeUsages() {
    super.removeUsages();
    symbols.forEach(symbol -> ((SymbolImpl) symbol).removeUsages());
  }

  @Override
  public Set<String> validForPythonVersions() {
    return alternatives().stream().flatMap(symbol -> ((SymbolImpl) symbol).validForPythonVersions().stream()).collect(Collectors.toSet());
  }
}
