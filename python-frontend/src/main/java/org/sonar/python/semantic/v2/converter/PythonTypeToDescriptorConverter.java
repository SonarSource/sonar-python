/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

import com.google.common.annotations.VisibleForTesting;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FullyQualifiedNameHelper;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.index.VariableDescriptor;

public class PythonTypeToDescriptorConverter {

  public Descriptor convert(String moduleFqn, SymbolV2 symbol, Set<PythonType> types) {
    var candidates = types.stream()
      .map(type -> convert(moduleFqn, moduleFqn, symbol.name(), type, symbol.usages()))
      .flatMap(candidate -> {
        if (candidate instanceof AmbiguousDescriptor ambiguousDescriptor) {
          return ambiguousDescriptor.alternatives().stream();
        } else {
          return Stream.of(candidate);
        }
      })
      .collect(Collectors.toSet());

    if (candidates.size() == 1) {
      return candidates.iterator().next();
    } else if (candidates.size() > 1) {
      return new AmbiguousDescriptor(symbol.name(), symbolFqn(moduleFqn, symbol.name()), candidates);
    }
    throw new IllegalStateException("No candidate found for descriptor " + symbol.name());
  }

  private static Descriptor convert(String moduleFqn, String parentFqn, String symbolName, PythonType type, List<UsageV2> symbolUsages) {
    if (type instanceof FunctionType functionType) {
      if (usagesContainAssignment(symbolUsages)) {
        // Avoid possible FPs in case of assigned bound method (SONARPY-2285)
        return new VariableDescriptor(symbolName, symbolFqn(parentFqn, symbolName), null);
      }
      return convert(functionType);
    }
    if (type instanceof SelfType selfType) {
      return convert(moduleFqn, parentFqn, symbolName, selfType);
    }
    if (type instanceof ClassType classType) {
      return convert(moduleFqn, parentFqn, symbolName, classType, false);
    }
    if (type instanceof UnionType unionType) {
      return convert(moduleFqn, parentFqn, symbolName, unionType);
    }
    if (type instanceof UnknownType.UnresolvedImportType unresolvedImportType) {
      return convert(parentFqn, symbolName, unresolvedImportType);
    }
    if (type instanceof ObjectType objectType && !moduleFqn.equals(parentFqn)) {
      return convert(parentFqn, symbolName, objectType);
    }
    return new VariableDescriptor(symbolName, symbolFqn(parentFqn, symbolName), null);
  }

  private static Descriptor convert(String parentFqn, String symbolName, ObjectType objectType) {
    return new VariableDescriptor(symbolName, symbolFqn(parentFqn, symbolName), FullyQualifiedNameHelper.getFullyQualifiedName(objectType.unwrappedType()).orElse(null));
  }

  private static Descriptor convert(FunctionType type) {

    var parameters = type.parameters()
      .stream()
      .map(PythonTypeToDescriptorConverter::convert)
      .toList();

    var decorators = type.decorators()
      .stream()
      .map(TypeWrapper::type)
      .map(decorator -> FullyQualifiedNameHelper.getFullyQualifiedName(decorator).orElse(null))
      .filter(Objects::nonNull)
      .toList();

    var returnType = type.returnType();
    var unwrappedReturnType = returnType.unwrappedType();
    var isSelfReturnType = containsSelfType(returnType);
    var annotatedReturnTypeName = FullyQualifiedNameHelper.getFullyQualifiedName(unwrappedReturnType).orElse(null);
    var returnTypeAnnotationDescriptor = createTypeAnnotationDescriptor(unwrappedReturnType, isSelfReturnType);

    // Using FunctionType#name and FunctionType#fullyQualifiedName instead of symbol is only accurate if the function has not been reassigned
    // This logic should be revisited when tackling SONARPY-2285
    return new FunctionDescriptor(type.name(), type.fullyQualifiedName(),
      parameters,
      type.isAsynchronous(),
      type.isInstanceMethod(),
      type.isClassMethod(),
      decorators,
      type.hasDecorators(),
      type.definitionLocation().orElse(null),
      annotatedReturnTypeName,
      returnTypeAnnotationDescriptor);
  }

  private static boolean containsSelfType(PythonType returnType){
    return returnType instanceof SelfType || returnType.unwrappedType() instanceof SelfType;
  }

  @VisibleForTesting
  static Descriptor convert(String moduleFqn, String parentFqn, String symbolName, SelfType selfType) {
    var innerType = selfType.innerType();
    if (!(innerType instanceof ClassType classType)) {
      throw new IllegalStateException("SelfType's innerType is not a ClassType " + selfType.name());
    }
    return convert(moduleFqn, parentFqn, symbolName, classType, true);
  }

  private static Descriptor convert(String moduleFqn, String parentFqn, String symbolName, ClassType type, boolean isSelf) {
    var symbolFqn = symbolFqn(parentFqn, symbolName);
    var memberDescriptors = type.members()
      .stream()
      .map(m -> convert(moduleFqn, symbolFqn, m.name(), m.type(), List.of()))
      .collect(Collectors.toSet());

    var hasSuperClassWithoutDescriptor = false;
    var superClasses = new ArrayList<String>();
    for (var superClassWrapper : type.superClasses()) {
      var superClassFqn = FullyQualifiedNameHelper.getFullyQualifiedName(superClassWrapper.type());
      if (superClassFqn.isPresent()) {
        superClasses.add(superClassFqn.get());
      } else {
        hasSuperClassWithoutDescriptor = true;
      }
    }

    var metaclassFQN = type.metaClasses()
      .stream()
      .map(metaClass -> FullyQualifiedNameHelper.getFullyQualifiedName(metaClass).orElse(null))
      .filter(Objects::nonNull)
      .findFirst()
      .orElse(null);

    return new ClassDescriptor.ClassDescriptorBuilder()
      .withName(symbolName)
      .withFullyQualifiedName(symbolFqn)
      .withSuperClasses(superClasses)
      .withMembers(memberDescriptors)
      .withHasDecorators(type.hasDecorators())
      .withDefinitionLocation(type.definitionLocation().orElse(null))
      .withHasSuperClassWithoutDescriptor(hasSuperClassWithoutDescriptor)
      .withHasMetaClass(type.hasMetaClass())
      .withMetaclassFQN(metaclassFQN)
      .withSupportsGenerics(type.isGeneric())
      .withIsSelf(isSelf)
      .build();
  }

  private static Descriptor convert(String moduleFqn, String parentFqn, String symbolName, UnionType type) {
    var candidates = type.candidates().stream()
      .map(candidateType -> convert(moduleFqn, parentFqn, symbolName, candidateType, List.of()))
      .collect(Collectors.toSet());
    return new AmbiguousDescriptor(symbolName,
      symbolFqn(moduleFqn, symbolName),
      candidates);
  }

  private static Descriptor convert(String parentFqn, String symbolName, UnknownType.UnresolvedImportType type) {
    return new VariableDescriptor(symbolName,
      symbolFqn(parentFqn, symbolName),
      type.importPath());
  }

  private static FunctionDescriptor.Parameter convert(ParameterV2 parameter) {
    var type = parameter.declaredType().type();
    var isSelf = containsSelfType(type);
    var unwrappedType = type.unwrappedType();
    var annotatedType = FullyQualifiedNameHelper.getFullyQualifiedName(unwrappedType).orElse(null);
    var typeAnnotationDescriptor = createTypeAnnotationDescriptor(unwrappedType, isSelf);

    return new FunctionDescriptor.Parameter(parameter.name(),
      annotatedType,
      typeAnnotationDescriptor,
      parameter.hasDefaultValue(),
      parameter.isKeywordOnly(),
      parameter.isPositionalOnly(),
      parameter.isPositionalVariadic(),
      parameter.isKeywordVariadic(),
      parameter.location());
  }

  private static String symbolFqn(String moduleFqn, String symbolName) {
    return moduleFqn + "." + symbolName;
  }

  private static boolean usagesContainAssignment(List<UsageV2> symbolUsages) {
    return symbolUsages.stream().anyMatch(u -> u.kind().equals(UsageV2.Kind.ASSIGNMENT_LHS));
  }

  @CheckForNull
  private static TypeAnnotationDescriptor createTypeAnnotationDescriptor(PythonType type, boolean isSelf) {
    if (type instanceof SelfType selfType) {
      return createTypeAnnotationDescriptor(selfType.innerType(), isSelf);
    }else if (type instanceof ClassType classType) {
      return new TypeAnnotationDescriptor(classType.name(), TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), classType.fullyQualifiedName(), isSelf);
    } else if (type instanceof FunctionType functionType) {
      return new TypeAnnotationDescriptor(functionType.name(), TypeAnnotationDescriptor.TypeKind.CALLABLE, List.of(), functionType.fullyQualifiedName(), false);
    } else if (type instanceof UnknownType.UnresolvedImportType importType) {
      return new TypeAnnotationDescriptor(importType.importPath(), TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), 
          FullyQualifiedNameHelper.getFullyQualifiedName(importType).orElse(null), false);
    }
    return null;
  }
}
