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
package org.sonar.python.semantic;

import java.util.Arrays;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Map;
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
    return copyWithoutUsages(new IdentityHashMap<>());
  }

  @Override
  AmbiguousSymbolImpl copyWithoutUsages(Map<SymbolImpl, SymbolImpl> copiedSymbols) {
    SymbolImpl existingCopy = copiedSymbols.get(this);
    if (existingCopy != null) {
      return (AmbiguousSymbolImpl) existingCopy;
    }
    Set<Symbol> copiedAlternativeSymbols = new HashSet<>();
    AmbiguousSymbolImpl copiedSymbol = new AmbiguousSymbolImpl(name(), fullyQualifiedName(), copiedAlternativeSymbols);
    // Register the incomplete copy before traversing alternatives so cyclic class hierarchies can refer back to it.
    copiedSymbols.put(this, copiedSymbol);
    flattenAmbiguousSymbols(symbols).stream()
      .map(SymbolImpl.class::cast)
      .map(symbol -> symbol.copyWithoutUsages(copiedSymbols))
      .forEach(copiedAlternativeSymbols::add);
    return copiedSymbol;
  }

  @Override
  public void removeUsages() {
    super.removeUsages();
    symbols.forEach(symbol -> ((SymbolImpl) symbol).removeUsages());
  }

  @Override
  public Set<String> validForSemanticPythonVersions() {
    return alternatives().stream().flatMap(symbol -> ((SymbolImpl) symbol).validForSemanticPythonVersions().stream()).collect(Collectors.toSet());
  }
}
