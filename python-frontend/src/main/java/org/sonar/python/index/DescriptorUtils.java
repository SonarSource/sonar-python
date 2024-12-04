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
package org.sonar.python.index;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.DeclaredType;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.semantic.SymbolUtils.typeshedSymbolWithFQN;
import static org.sonar.python.types.InferredTypes.anyType;

public class DescriptorUtils {

  private DescriptorUtils() {}

  // TODO SONARPY-958: Cleanup the symbol construction from descriptors by extracting this logic in a builder class
  public static Symbol symbolFromDescriptor(Descriptor descriptor, ProjectLevelSymbolTable projectLevelSymbolTable,
                                            @Nullable String localSymbolName, Map<Descriptor, Symbol> createdSymbolsByDescriptor, Map<String, Symbol> createdSymbolsByFqn) {
    if (createdSymbolsByDescriptor.containsKey(descriptor)) {
      return createdSymbolsByDescriptor.get(descriptor);
    } else if (descriptor.fullyQualifiedName() != null && createdSymbolsByFqn.containsKey(descriptor.fullyQualifiedName())) {
      return createdSymbolsByFqn.get(descriptor.fullyQualifiedName());
    }
    // The symbol generated from the descriptor will not have the descriptor name if an alias (localSymbolName) is defined
    String symbolName = localSymbolName != null ? localSymbolName : descriptor.name();
    switch (descriptor.kind()) {
      case CLASS:
        return createClassSymbol(descriptor, projectLevelSymbolTable, createdSymbolsByDescriptor, createdSymbolsByFqn, symbolName);
      case FUNCTION:
        return createFunctionSymbol((FunctionDescriptor) descriptor, projectLevelSymbolTable, createdSymbolsByDescriptor, createdSymbolsByFqn, symbolName);
      case VARIABLE:
        var variableDescriptor = (VariableDescriptor) descriptor;
        return new SymbolImpl(symbolName, descriptor.fullyQualifiedName(), variableDescriptor.annotatedType());
      case AMBIGUOUS:
        Set<Symbol> alternatives = new HashSet<>();
        AmbiguousSymbolImpl ambiguousSymbol = new AmbiguousSymbolImpl(symbolName, descriptor.fullyQualifiedName(), alternatives);
        createdSymbolsByDescriptor.put(descriptor, ambiguousSymbol);
        alternatives.addAll(((AmbiguousDescriptor) descriptor).alternatives().stream()
          // Alternatives of ambiguous descriptors share the same FQN despite representing different possible symbols, so we don't rely on "createdSymbolsByFqn" for them
          .map(a -> DescriptorUtils.symbolFromDescriptor(a, projectLevelSymbolTable, symbolName, createdSymbolsByDescriptor, new HashMap<>()))
          .collect(Collectors.toSet()));
        return ambiguousSymbol;
      default:
        throw new IllegalStateException(String.format("Error while creating a Symbol from a Descriptor: Unexpected descriptor kind: %s", descriptor.kind()));
    }
  }

  private static ClassSymbolImpl createClassSymbol(Descriptor descriptor, ProjectLevelSymbolTable projectLevelSymbolTable, Map<Descriptor, Symbol> createdSymbolsByDescriptor,
    Map<String, Symbol> createdSymbolByFqn, String symbolName) {
    ClassDescriptor classDescriptor = (ClassDescriptor) descriptor;
    ClassSymbolImpl classSymbol = new ClassSymbolImpl((ClassDescriptor) descriptor, symbolName);
    createdSymbolsByDescriptor.put(descriptor, classSymbol);
    createdSymbolByFqn.put(descriptor.fullyQualifiedName(), classSymbol);
    addSuperClasses(classSymbol, classDescriptor, projectLevelSymbolTable, createdSymbolsByDescriptor, createdSymbolByFqn);
    addMembers(classSymbol, classDescriptor, projectLevelSymbolTable, createdSymbolsByDescriptor, createdSymbolByFqn);
    return classSymbol;
  }

  private static void addMembers(ClassSymbolImpl classSymbol, ClassDescriptor classDescriptor,
                                 ProjectLevelSymbolTable projectLevelSymbolTable, Map<Descriptor, Symbol> createdSymbolsByDescriptor,
    Map<String , Symbol> createdSymbolsByFqn) {
    classSymbol.addMembers(classDescriptor.members().stream()
      .map(memberFqn -> DescriptorUtils.symbolFromDescriptor(memberFqn, projectLevelSymbolTable, null, createdSymbolsByDescriptor, createdSymbolsByFqn))
      .map(member -> {
        if (member instanceof FunctionSymbolImpl functionSymbol) {
          functionSymbol.setOwner(classSymbol);
        }
        return member;
      })
      .toList());
  }

  private static void addSuperClasses(ClassSymbolImpl classSymbol, ClassDescriptor classDescriptor,
                                      ProjectLevelSymbolTable projectLevelSymbolTable, Map<Descriptor, Symbol> createdSymbolsByDescriptor,
    Map<String, Symbol> createdSymbolsByFqn) {
    classDescriptor.superClasses().stream()
      .map(superClassFqn -> {
          if (createdSymbolsByFqn.containsKey(superClassFqn)) {
            return createdSymbolsByFqn.get(superClassFqn);
          }
          Symbol symbol = projectLevelSymbolTable.getSymbol(superClassFqn, null, createdSymbolsByDescriptor, createdSymbolsByFqn);
          symbol = symbol != null ? symbol : typeshedSymbolWithFQN(superClassFqn);
          createdSymbolsByFqn.put(superClassFqn, symbol);
          return symbol;
        }
      )
      .forEach(classSymbol::addSuperClass);
  }

  private static FunctionSymbolImpl createFunctionSymbol(FunctionDescriptor functionDescriptor, ProjectLevelSymbolTable projectLevelSymbolTable,
                                                         Map<Descriptor, Symbol> createdSymbolsByDescriptor, Map<String, Symbol> createdSymbolsByFqn,
                                                         String symbolName) {
    FunctionSymbolImpl functionSymbol = new FunctionSymbolImpl(functionDescriptor, symbolName);
    addParameters(functionSymbol, functionDescriptor, projectLevelSymbolTable, createdSymbolsByDescriptor, createdSymbolsByFqn);
    return functionSymbol;
  }

  private static void addParameters(FunctionSymbolImpl functionSymbol, FunctionDescriptor functionDescriptor,
                                    ProjectLevelSymbolTable projectLevelSymbolTable, Map<Descriptor, Symbol> createdSymbolsByDescriptor, Map<String, Symbol> createdSymbolsByFqn) {
    functionDescriptor.parameters().stream().map(parameterDescriptor -> {
      FunctionSymbolImpl.ParameterImpl parameter = new FunctionSymbolImpl.ParameterImpl(parameterDescriptor);
      setParameterType(parameter, parameterDescriptor.annotatedType(), projectLevelSymbolTable, createdSymbolsByDescriptor, createdSymbolsByFqn);
      return parameter;
    }).forEach(functionSymbol::addParameter);
  }

  private static void setParameterType(FunctionSymbolImpl.ParameterImpl parameter, String annotatedType, ProjectLevelSymbolTable projectLevelSymbolTable,
                                       Map<Descriptor, Symbol> createdSymbolsByDescriptor, Map<String, Symbol> createdSymbolsByFqn) {
    InferredType declaredType;
    if (parameter.isKeywordVariadic()) {
      declaredType = InferredTypes.DICT;
    } else if (parameter.isPositionalVariadic()) {
      declaredType = InferredTypes.TUPLE;
    } else {
      Symbol existingSymbol = createdSymbolsByFqn.get(annotatedType);
      Symbol typeSymbol = existingSymbol != null ? existingSymbol : projectLevelSymbolTable.getSymbol(annotatedType, null, createdSymbolsByDescriptor, createdSymbolsByFqn);
      String annotatedTypeName = parameter.annotatedTypeName();
      if (typeSymbol == null && annotatedTypeName != null) {
        typeSymbol = typeshedSymbolWithFQN(annotatedTypeName);
      }
      declaredType = typeSymbol == null ? anyType() : new DeclaredType(typeSymbol, Collections.emptyList());
    }
    parameter.setDeclaredType(declaredType);
  }
}
