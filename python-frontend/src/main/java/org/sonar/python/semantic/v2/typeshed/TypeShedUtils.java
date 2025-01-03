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
package org.sonar.python.semantic.v2.typeshed;

import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class TypeShedUtils {

  private TypeShedUtils() {
  }

  public static String normalizedFqn(String fqn) {
    if (fqn.isEmpty()) {
      return null;
    }
    if (fqn.startsWith(TypeShedConstants.BUILTINS_PREFIX)) {
      return fqn.substring(TypeShedConstants.BUILTINS_PREFIX.length());
    }
    return fqn;
  }

  @CheckForNull
  public static String getTypesNormalizedFqn(@Nullable SymbolsProtos.Type type) {
    return Optional.ofNullable(type)
      .map(TypeShedUtils::getTypesFqn)
      .map(TypeShedUtils::normalizedFqn)
      .orElse(null);
  }

  @CheckForNull
  private static String getTypesFqn(SymbolsProtos.Type type) {
    // Add support CALLABLE and UNION kinds
    switch (type.getKind()) {
      case INSTANCE:
        String typeName = type.getFullyQualifiedName();
        return typeName.isEmpty() ? null : typeName;
      case TYPE_ALIAS:
        return getTypesFqn(type.getArgs(0));
      case TYPE:
        return TypeShedConstants.BUILTINS_TYPE_FQN;
      case TUPLE:
        return TypeShedConstants.BUILTINS_TUPLE_FQN;
      case NONE:
        return TypeShedConstants.BUILTINS_NONE_TYPE_FQN;
      case TYPED_DICT:
        return TypeShedConstants.BUILTINS_DICT_FQN;
      case TYPE_VAR:
        return Optional.of(type)
          .filter(TypeShedUtils::filterTypeVar)
          .map(SymbolsProtos.Type::getFullyQualifiedName)
          .orElse(null);
      default:
        return null;
    }
  }

  // ref: SONARPY-1477
  private static final Set<String> EXCLUDING_TYPE_VAR_FQN_PATTERNS = Set.of(
    "^builtins\\.object$",
    "^_ctypes\\._CanCastTo$");

  public static boolean filterTypeVar(SymbolsProtos.Type type) {
    return Optional.of(type)
      // Filtering self returning methods until the SONARPY-1472 will be solved
      .filter(Predicate.not(t -> t.getPrettyPrintedName().endsWith(".Self")))
      .map(SymbolsProtos.Type::getFullyQualifiedName)
      .filter(Predicate.not(String::isEmpty))
      .filter(fqn -> EXCLUDING_TYPE_VAR_FQN_PATTERNS.stream().noneMatch(fqn::matches))
      .isPresent();
  }

}
