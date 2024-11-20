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
package org.sonar.python.semantic.v2.converter;

import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.types.v2.LazyUnionType;
import org.sonar.python.types.v2.PythonType;

public class TypeAnnotationToPythonTypeConverter {

  public PythonType convert(ConversionContext context, TypeAnnotationDescriptor type) {
    switch (type.kind()) {
      case INSTANCE:
        String fullyQualifiedName = type.fullyQualifiedName();
        if (fullyQualifiedName == null) {
          return PythonType.UNKNOWN;
        }
        // _SpecialForm is the type used for some special types, like Callable, Union, TypeVar, ...
        // It comes from CPython impl: https://github.com/python/cpython/blob/e39ae6bef2c357a88e232dcab2e4b4c0f367544b/Lib/typing.py#L439
        // This doesn't seem to be very precisely specified in typeshed, because it has special semantic.
        // To avoid FPs, we treat it as ANY
        if ("typing._SpecialForm".equals(fullyQualifiedName)) {
          return PythonType.UNKNOWN;
        }
        return fullyQualifiedName.isEmpty() ? PythonType.UNKNOWN : context.lazyTypesContext().getOrCreateLazyType(fullyQualifiedName);
      case TYPE:
        return context.lazyTypesContext().getOrCreateLazyType("type");
      case TYPE_ALIAS:
        return convert(context, type.args().get(0));
      case CALLABLE:
        // this should be handled as a function type - see SONARPY-953
        return context.lazyTypesContext().getOrCreateLazyType("function");
      case UNION:
        return new LazyUnionType(type.args().stream().map(t -> convert(context, t)).collect(Collectors.toSet()));
      case TUPLE:
        return context.lazyTypesContext().getOrCreateLazyType("tuple");
      case NONE:
        return context.lazyTypesContext().getOrCreateLazyType("NoneType");
      case TYPED_DICT:
        // SONARPY-2179: This case only makes sense for parameter types, which are not supported yet
        return context.lazyTypesContext().getOrCreateLazyType("dict");
      case TYPE_VAR:
        return Optional.of(type)
          .filter(TypeAnnotationToPythonTypeConverter::filterTypeVar)
          .map(TypeAnnotationDescriptor::fullyQualifiedName)
          .map(context.lazyTypesContext()::getOrCreateLazyType)
          .map(PythonType.class::cast)
          .orElse(PythonType.UNKNOWN);
      default:
        return PythonType.UNKNOWN;
    }
  }

  private static final Set<String> EXCLUDING_TYPE_VAR_FQN_PATTERNS = Set.of(
    "object",
    "^builtins\\.object$",
    // ref: SONARPY-1477
    "^_ctypes\\._CanCastTo$");

  public static boolean filterTypeVar(TypeAnnotationDescriptor type) {
    return Optional.of(type)
      // Filtering self returning methods until the SONARPY-1472 will be solved
      .filter(Predicate.not(t -> t.prettyPrintedName().endsWith(".Self")))
      .map(TypeAnnotationDescriptor::fullyQualifiedName)
      .filter(Predicate.not(String::isEmpty))
      // We ignore TypeVar referencing "builtins.object" or "object" to avoid false positives
      .filter(fqn -> EXCLUDING_TYPE_VAR_FQN_PATTERNS.stream().noneMatch(fqn::matches))
      .isPresent();
  }
}
