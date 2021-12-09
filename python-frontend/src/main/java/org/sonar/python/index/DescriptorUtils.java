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
package org.sonar.python.index;

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolImpl;

public class DescriptorUtils {

  private DescriptorUtils() {}


  public static Collection<Descriptor> descriptors(Symbol symbol) {
    switch (symbol.kind()) {
      case FUNCTION:
        return Collections.singleton(functionDescriptor(((FunctionSymbol) symbol)));
      case CLASS:
        return Collections.singleton(classDescriptor((ClassSymbol) symbol));
      case AMBIGUOUS:
        return ((AmbiguousSymbol) symbol).alternatives().stream().flatMap(s -> descriptors(s).stream()).collect(Collectors.toSet());
      default:
        return Collections.singleton(new VariableDescriptor(symbol.name(), symbol.fullyQualifiedName(), symbol.annotatedTypeName()));
    }
  }

  private static ClassDescriptor classDescriptor(ClassSymbol classSymbol) {
    ClassDescriptor.ClassDescriptorBuilder classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName(classSymbol.name())
      .withFullyQualifiedName(classSymbol.fullyQualifiedName())
      .withMembers(classSymbol.declaredMembers().stream().flatMap(s -> descriptors(s).stream()).collect(Collectors.toList()))
      .withSuperClasses(classSymbol.superClasses().stream().map(Symbol::fullyQualifiedName).filter(Objects::nonNull).collect(Collectors.toList()))
      .withDefinitionLocation(classSymbol.definitionLocation())
      .withHasMetaClass(((ClassSymbolImpl) classSymbol).hasMetaClass())
      .withHasSuperClassWithoutDescriptor(((ClassSymbolImpl) classSymbol).hasSuperClassWithoutSymbol() ||
        classSymbol.superClasses().stream().anyMatch(s -> s.fullyQualifiedName() == null))
      .withMetaclassFQN(((ClassSymbolImpl) classSymbol).metaclassFQN())
      .withHasDecorators(classSymbol.hasDecorators())
      .withSupportsGenerics(((ClassSymbolImpl) classSymbol).supportsGenerics());

    return classDescriptor.build();
  }

  private static FunctionDescriptor functionDescriptor(FunctionSymbol functionSymbol) {
    return new FunctionDescriptor.FunctionDescriptorBuilder()
      .withName(functionSymbol.name())
      .withFullyQualifiedName(functionSymbol.fullyQualifiedName())
      .withParameters(parameters(functionSymbol.parameters()))
      .withHasDecorators(functionSymbol.hasDecorators())
      .withDecorators(functionSymbol.decorators())
      .withIsAsynchronous(functionSymbol.isAsynchronous())
      .withIsInstanceMethod(functionSymbol.isInstanceMethod())
      .withAnnotatedReturnTypeName(functionSymbol.annotatedReturnTypeName())
      .withDefinitionLocation(functionSymbol.definitionLocation())
      .build();
  }

  private static List<FunctionDescriptor.Parameter> parameters(List<FunctionSymbol.Parameter> parameters) {
    return parameters.stream().map(parameter -> new FunctionDescriptor.Parameter(
      parameter.name(),
      ((FunctionSymbolImpl.ParameterImpl) parameter).annotatedTypeName(),
      parameter.hasDefaultValue(),
      parameter.isVariadic(),
      parameter.isKeywordOnly(),
      parameter.isPositionalOnly(),
      parameter.location()
    )).collect(Collectors.toList());
  }


  public static Symbol symbolFromDescriptor(Descriptor descriptor, ProjectLevelSymbolTable projectLevelSymbolTable) {
    return symbolFromDescriptor(descriptor, projectLevelSymbolTable, null);
  }

  public static Symbol symbolFromDescriptor(Descriptor descriptor, ProjectLevelSymbolTable projectLevelSymbolTable, @Nullable String localSymbolName) {
    // The symbol generated from the descriptor will not have the descriptor name if an alias (localSymbolName) is defined
    String symbolName = localSymbolName != null ? localSymbolName : descriptor.name();
    switch (descriptor.kind()) {
      case CLASS:
        return classSymbol((ClassDescriptor) descriptor, projectLevelSymbolTable, symbolName);
      case FUNCTION:
        return functionSymbol((FunctionDescriptor) descriptor, projectLevelSymbolTable, symbolName);
      case VARIABLE:
        return new SymbolImpl(symbolName, descriptor.fullyQualifiedName());
      default:
        // Ambiguous descriptor
        return ambiguousSymbol((AmbiguousDescriptor) descriptor, projectLevelSymbolTable, symbolName);
    }
  }

  private static ClassSymbol classSymbol(ClassDescriptor classDescriptor, ProjectLevelSymbolTable projectLevelSymbolTable, String symbolName) {
    return new ClassSymbolImpl(classDescriptor, projectLevelSymbolTable, symbolName);
  }

  private static FunctionSymbol functionSymbol(FunctionDescriptor functionDescriptor, ProjectLevelSymbolTable projectLevelSymbolTable, String symbolName) {
    return new FunctionSymbolImpl(functionDescriptor, projectLevelSymbolTable, symbolName);
  }

  public static FunctionSymbol.Parameter functionParameter(FunctionDescriptor.Parameter parameterDescriptor, ProjectLevelSymbolTable projectLevelSymbolTable) {
    return new FunctionSymbolImpl.ParameterImpl(parameterDescriptor, projectLevelSymbolTable);
  }

  public static AmbiguousSymbol ambiguousSymbol(AmbiguousDescriptor ambiguousDescriptor, ProjectLevelSymbolTable projectLevelSymbolTable, String symbolName) {
    Set<Symbol> descriptors = ambiguousDescriptor.alternatives().stream()
      .map(a -> DescriptorUtils.symbolFromDescriptor(a, projectLevelSymbolTable, symbolName))
      .collect(Collectors.toSet());
    return new AmbiguousSymbolImpl(symbolName, ambiguousDescriptor.fullyQualifiedName(), descriptors);
  }
}
