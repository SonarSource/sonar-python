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
package org.sonar.python.types;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.SymbolImpl;

/**
 * This type is used for symbols that we assume to be class due to how they are used,
 * so that type inference correctly tracks objects that were created from this.
 */
public class UnknownClassType implements InferredType {

  private final Symbol typeSymbol;
  private final Map<String, Symbol> members;

  public UnknownClassType(Symbol typeSymbol) {
    this.typeSymbol = typeSymbol;
    this.members = new HashMap<>();
  }

  @Override
  public boolean isIdentityComparableWith(InferredType other) {
    return true;
  }

  @Override
  public boolean canHaveMember(String memberName) {
    return true;
  }

  @Override
  public boolean declaresMember(String memberName) {
    return true;
  }

  @Override
  public Optional<Symbol> resolveMember(String memberName) {
    var member = members.computeIfAbsent(memberName, n -> Optional.of(typeSymbol)
      .map(Symbol::fullyQualifiedName)
      .map(fqn -> new SymbolImpl(memberName, fqn + "." + memberName))
      .orElse(null));
    return Optional.ofNullable(member);
  }

  @Override
  public Optional<Symbol> resolveDeclaredMember(String memberName) {
    return resolveMember(memberName);
  }

  @Override
  public boolean canOnlyBe(String typeName) {
    return false;
  }

  @Override
  public boolean canBeOrExtend(String typeName) {
    return true;
  }

  @Override
  public boolean isCompatibleWith(InferredType other) {
    return true;
  }

  @Override
  public boolean mustBeOrExtend(String typeName) {
    return false;
  }

  public Symbol typeSymbol() {
    return typeSymbol;
  }
}
