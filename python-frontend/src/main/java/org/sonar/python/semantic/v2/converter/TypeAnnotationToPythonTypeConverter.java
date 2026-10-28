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
package org.sonar.python.semantic.v2.converter;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.types.v2.LazyUnionType;

public class TypeAnnotationToPythonTypeConverter {

  // Annotations describe runtime values, including when nested in generic arguments.
  // Keep class resolution lazy; unions resolve their value wrappers before hashing them.
  public PythonType convert(ConversionContext context, TypeAnnotationDescriptor type) {
    switch (type.kind()) {
      case INSTANCE:
        String fullyQualifiedName = type.fullyQualifiedName();
        if (fullyQualifiedName == null) {
          return PythonType.UNKNOWN;
        }
        List<PythonType> attributes = type.args().stream().map(t -> convert(context, t)).toList();

        if (Boolean.TRUE.equals(type.isSelf())) {
          var selfType = SelfType.fromTypeWrapper(TypeWrapper.of(typeFromFQN(fullyQualifiedName, context)));
          return ObjectType.Builder.fromType(selfType)
            .withAttributes(attributes)
            .build();
        }
        // _SpecialForm is the type used for some special types, like Callable, Union, TypeVar, ...
        // It comes from CPython impl: https://github.com/python/cpython/blob/e39ae6bef2c357a88e232dcab2e4b4c0f367544b/Lib/typing.py#L439
        // This doesn't seem to be very precisely specified in typeshed, because it has special semantic.
        // To avoid FPs, we treat it as ANY
        if ("typing._SpecialForm".equals(fullyQualifiedName)) {
          return PythonType.UNKNOWN;
        }
        return ObjectType.Builder.fromType(typeFromFQN(fullyQualifiedName, context))
          .withAttributes(attributes)
          .build();
      case TYPE:
        return ObjectType.fromType(context.lazyTypesContext().getOrCreateLazyType("type"));
      case TYPE_ALIAS:
        return convert(context, type.args().get(0));
      case CALLABLE:
        // this should be handled as a function type - see SONARPY-953
        return ObjectType.fromType(context.lazyTypesContext().getOrCreateLazyType("function"));
      case UNION:
        return LazyUnionType.or(type.args().stream().map(t -> convert(context, t)).collect(Collectors.toSet()));
      case TUPLE:
        return ObjectType.fromType(context.lazyTypesContext().getOrCreateLazyType("tuple"));
      case NONE:
        return ObjectType.fromType(context.lazyTypesContext().getOrCreateLazyType("NoneType"));
      case TYPED_DICT:
        // SONARPY-2179: This case only makes sense for parameter types, which are not supported yet
        return ObjectType.fromType(context.lazyTypesContext().getOrCreateLazyType("dict"));
      case TYPE_VAR:
        if (Boolean.TRUE.equals(type.isSelf())) {
          if (type.args().size() != 1) {
            return PythonType.UNKNOWN;
          }
          PythonType innerType = convert(context, type.args().get(0));
          if (innerType instanceof ObjectType objectType) {
            return ObjectType.Builder.fromType(objectType)
              .withType(SelfType.fromTypeWrapper(objectType.typeWrapper()))
              .build();
          }
          return PythonType.UNKNOWN;
        }
        return Optional.of(type)
          .filter(TypeAnnotationToPythonTypeConverter::filterTypeVar)
          .map(TypeAnnotationDescriptor::fullyQualifiedName)
          .map(context.lazyTypesContext()::getOrCreateLazyType)
          .<PythonType>map(ObjectType::fromType)
          .orElse(PythonType.UNKNOWN);
      default:
        return PythonType.UNKNOWN;
    }
  }

  PythonType convertReturnType(ConversionContext context, TypeAnnotationDescriptor type) {
    // Preserve bottom in return annotations, consumed by call and property inference.
    // Non-return annotations and generic arguments retain their usual conversion.
    return switch (type.kind()) {
      case UNINHABITED -> PythonType.NEVER;
      case UNION -> LazyUnionType.or(type.args().stream().map(t -> convertReturnType(context, t)).collect(Collectors.toSet()));
      case TYPE_ALIAS -> convertReturnType(context, type.args().get(0));
      default -> convert(context, type);
    };
  }

  private static PythonType typeFromFQN(String fullyQualifiedName, ConversionContext context) {
    return fullyQualifiedName.isEmpty() ? PythonType.UNKNOWN : context.lazyTypesContext().getOrCreateLazyType(fullyQualifiedName);
  }

  private static final Set<String> EXCLUDED_TYPE_VAR_FQNS = Set.of(
    "object",
    "builtins.object",
    // ref: SONARPY-1477
    "_ctypes._CanCastTo");

  public static boolean filterTypeVar(TypeAnnotationDescriptor type) {
    // Ignore these exact bounds to avoid false positives (SONARPY-1472, SONARPY-1477).
    var fqn = type.fullyQualifiedName();
    return fqn != null && !fqn.isEmpty() && !EXCLUDED_TYPE_VAR_FQNS.contains(fqn);
  }
}
