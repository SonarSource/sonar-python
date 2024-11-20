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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeWrapper;
import org.sonar.python.types.v2.UnionType;
import org.sonar.python.types.v2.UnknownType;

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
      return convert(moduleFqn, functionType);
    }
    if (type instanceof ClassType classType) {
      return convert(moduleFqn, parentFqn, symbolName, classType);
    }
    if (type instanceof UnionType unionType) {
      return convert(moduleFqn, parentFqn, symbolName, unionType);
    }
    if (type instanceof UnknownType.UnresolvedImportType unresolvedImportType) {
      return convert(parentFqn, symbolName, unresolvedImportType);
    }
    if (type instanceof ObjectType objectType && !moduleFqn.equals(parentFqn)) {
      return convert(moduleFqn, parentFqn, symbolName, objectType);
    }
    return new VariableDescriptor(symbolName, symbolFqn(parentFqn, symbolName), null);
  }

  private static Descriptor convert(String moduleFqn, String parentFqn, String symbolName, ObjectType objectType) {
    return new VariableDescriptor(symbolName, symbolFqn(parentFqn, symbolName), typeFqn(moduleFqn, objectType.unwrappedType()));
  }

  private static Descriptor convert(String moduleFqn, FunctionType type) {

    var parameters = type.parameters()
      .stream()
      .map(parameter -> convert(moduleFqn, parameter))
      .toList();

    var decorators = type.decorators()
      .stream()
      .map(TypeWrapper::type)
      .map(decorator -> typeFqn(moduleFqn, decorator))
      .filter(Objects::nonNull)
      .toList();

    // Using FunctionType#name and FunctionType#fullyQualifiedName instead of symbol is only accurate if the function has not been reassigned
    // This logic should be revisited when tackling SONARPY-2285
    return new FunctionDescriptor(type.name(), type.fullyQualifiedName(),
      parameters,
      type.isAsynchronous(),
      type.isInstanceMethod(),
      decorators,
      type.hasDecorators(),
      type.definitionLocation().orElse(null),
      null,
      null
    );
  }

  private static Descriptor convert(String moduleFqn, String parentFqn, String symbolName, ClassType type) {
    var symbolFqn = symbolFqn(parentFqn, symbolName);
    var memberDescriptors = type.members()
      .stream()
      .map(m -> convert(moduleFqn, symbolFqn, m.name(), m.type(), List.of()))
      .collect(Collectors.toSet());

    var hasSuperClassWithoutDescriptor = false;
    var superClasses = new ArrayList<String>();
    for (var superClassWrapper : type.superClasses()) {
      var superClassFqn = typeFqn(moduleFqn, superClassWrapper.type());
      if (superClassFqn != null) {
        superClasses.add(superClassFqn);
      } else {
        hasSuperClassWithoutDescriptor = true;
      }
    }

    var metaclassFQN = type.metaClasses()
      .stream()
      .map(metaClass -> typeFqn(moduleFqn, metaClass))
      .filter(Objects::nonNull)
      .findFirst()
      .orElse(null);

    return new ClassDescriptor(symbolName, symbolFqn,
      superClasses,
      memberDescriptors,
      type.hasDecorators(),
      type.definitionLocation().orElse(null),
      hasSuperClassWithoutDescriptor,
      type.hasMetaClass(),
      metaclassFQN,
      type.isGeneric()
    );
  }

  private static Descriptor convert(String moduleFqn, String parentFqn, String symbolName, UnionType type) {
    var candidates = type.candidates().stream()
      .map(candidateType -> convert(moduleFqn, parentFqn, symbolName, candidateType, List.of()))
      .collect(Collectors.toSet());
    return new AmbiguousDescriptor(symbolName,
      symbolFqn(moduleFqn, symbolName),
      candidates
    );
  }

  private static Descriptor convert(String parentFqn, String symbolName, UnknownType.UnresolvedImportType type) {
    return new VariableDescriptor(symbolName,
      symbolFqn(parentFqn, symbolName),
      type.importPath()
    );
  }

  private static FunctionDescriptor.Parameter convert(String moduleFqn, ParameterV2 parameter) {
    var type = parameter.declaredType().type().unwrappedType();
    var annotatedType = typeFqn(moduleFqn, type);

    return new FunctionDescriptor.Parameter(parameter.name(),
      annotatedType,
      parameter.hasDefaultValue(),
      parameter.isKeywordOnly(),
      parameter.isPositionalOnly(),
      parameter.isPositionalVariadic(),
      parameter.isKeywordVariadic(),
      parameter.location());
  }

  @CheckForNull
  private static String typeFqn(String moduleFqn, PythonType type) {
    if (type instanceof UnknownType.UnresolvedImportType importType) {
      return importType.importPath();
    } else if (type instanceof ClassType classType) {
      return classType.fullyQualifiedName();
    } else if (type instanceof FunctionType functionType) {
      return functionType.fullyQualifiedName();
    }
    return null;
  }

  private static String symbolFqn(String moduleFqn, String symbolName) {
    return moduleFqn + "." + symbolName;
  }

  private static boolean usagesContainAssignment(List<UsageV2> symbolUsages) {
    return symbolUsages.stream().anyMatch(u -> u.kind().equals(UsageV2.Kind.ASSIGNMENT_LHS));
  }
}
