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

import java.util.Optional;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;

enum AnyType implements InferredType {
  ANY;

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
    return Optional.empty();
  }

  @Override
  public Optional<Symbol> resolveDeclaredMember(String memberName) {
    return Optional.empty();
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
}
