/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python.api.types;

import com.google.common.annotations.Beta;
import java.util.Optional;
import java.util.Set;

import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;

@Beta
public interface InferredType {

  Set<String> MOCK_FQNS = Set.of("unittest.mock.Mock", "unittest.mock.MagicMock");

  @Beta
  boolean isIdentityComparableWith(InferredType other);

  @Beta
  boolean canHaveMember(String memberName);

  /**
   * Used to handle uncertainty of declared types. It return false when a member is not present in a declared type,
   * while canHaveMember returns true because one of its subtype may have it.
   */
  @Beta
  boolean declaresMember(String memberName);

  @Beta
  Optional<Symbol> resolveMember(String memberName);

  @Beta
  Optional<Symbol> resolveDeclaredMember(String memberName);

  @Beta
  boolean canOnlyBe(String typeName);

  @Beta
  boolean canBeOrExtend(String typeName);

  @Beta
  boolean isCompatibleWith(InferredType other);

  /**
   * mustBeOrExtend implies we know for sure the given type is either of the given typeName or a subtype of it.
   * As opposed to "canBeOrExtend", this will return true only when we are sure the subtyping relationship is present.
   * For types inferred from type annotations (DeclaredType), the actual underlying type might be different from what has been declared,
   * but it must be or extend the declared type.
   */
  @Beta
  boolean mustBeOrExtend(String typeName);

  /**
   * Returns type ClassSymbol only in case of RuntimeType.
   */
  @Beta
  @CheckForNull
  default ClassSymbol runtimeTypeSymbol() {
    return null;
  }

}
