/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic.v2;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

public class SymbolsModuleTypeProvider {
  private final ProjectLevelSymbolTable projectLevelSymbolTable;
  private final TypeShed typeShed;
  private ModuleType rootModule;
  private LazyTypesContext lazyTypesContext;

  public SymbolsModuleTypeProvider(ProjectLevelSymbolTable projectLevelSymbolTable, TypeShed typeShed) {
    this.projectLevelSymbolTable = projectLevelSymbolTable;
    this.typeShed = typeShed;
    this.lazyTypesContext = new LazyTypesContext();
    this.rootModule = createModuleFromSymbols(null, null, typeShed.builtinSymbols().values());
  }

  public ModuleType createBuiltinModule() {
    return rootModule;
  }

  public ModuleType createModuleType(List<String> moduleFqn, ModuleType parent) {
    var moduleName = moduleFqn.get(moduleFqn.size() - 1);
    var moduleFqnString = getModuleFqnString(moduleFqn);
    return createModuleTypeFromProjectLevelSymbolTable(moduleName, moduleFqnString, parent)
      .or(() -> createModuleTypeFromTypeShed(moduleName, moduleFqnString, parent))
      .orElseGet(() -> createEmptyModule(moduleName, parent));
  }

  PythonType resolveLazyTypeForFqn(String fullyQualifiedName) {
    String[] split = fullyQualifiedName.split("\\.");
    ModuleType currentModule = rootModule;
    for (int i = 0; i < split.length; i++) {
      String currentElement = split[i];
      Optional<PythonType> pythonType = currentModule.resolveMember(currentElement);
      if (pythonType.isEmpty()) {
        // We should probably return an "UnresolvedType" here, keeping the FQN available for possible checks
        return PythonType.UNKNOWN;
      }
      PythonType currentType = pythonType.get();
      if (i == split.length - 1) {
        // We have resolved the type we were looking for, return it
        return currentType;
      }
      if (!(currentType instanceof ModuleType moduleType)) {
        break;
      }
      // We have resolved an enclosing module, continue searching
      currentModule = moduleType;
    }
    return PythonType.UNKNOWN;
  }

  public PythonType resolveLazyType(LazyType lazyType) {
    if (lazyTypesContext.resolvedLazyTypes().containsKey(lazyType)) {
      return lazyTypesContext.resolvedLazyTypes().get(lazyType);
    }
    Queue<String> modules = lazyTypesContext.lazyTypeRequirements().get(lazyType);
    ModuleType parent = rootModule;
    while (!modules.isEmpty()) {
      String nextModule = modules.poll();
      // We shouldn't create a new module again if it has been resolved since
      Optional<PythonType> pythonType = parent.resolveMember(nextModule);
      if (pythonType.isEmpty()) {
        List<String> fqnParts = Arrays.stream(nextModule.split("\\.")).toList();
        parent = createModuleType(fqnParts, parent);
        continue;
      }
      parent = (ModuleType) pythonType.get();
    }
    PythonType resolvedType = resolveLazyTypeForFqn(lazyType.fullyQualifiedName());
    lazyTypesContext.addResolvedLazyType(lazyType, resolvedType);
    return resolvedType;
  }

  private static String getModuleFqnString(List<String> moduleFqn) {
    return String.join(".", moduleFqn);
  }

  private Optional<ModuleType> createModuleTypeFromProjectLevelSymbolTable(String moduleName, String moduleFqn, ModuleType parent) {
    return Optional.ofNullable(projectLevelSymbolTable.getSymbolsFromModule(moduleFqn))
      .map(projectModuleSymbols -> createModuleFromSymbols(moduleName, parent, projectModuleSymbols));
  }

  private Optional<ModuleType> createModuleTypeFromTypeShed(String moduleName, String moduleFqn, ModuleType parent) {
    return Optional.ofNullable(typeShed.symbolsForModule(moduleFqn))
      .filter(Predicate.not(Map::isEmpty))
      .map(typeShedModuleSymbols -> createModuleFromSymbols(moduleName, parent, typeShedModuleSymbols.values()));
  }

  private static ModuleType createEmptyModule(String moduleName, ModuleType parent) {
    var emptyModule = new ModuleType(moduleName, parent);
    parent.members().put(moduleName, emptyModule);
    return emptyModule;
  }

  private ModuleType createModuleFromSymbols(@Nullable String name, @Nullable ModuleType parent, Collection<Symbol> symbols) {
    var members = new HashMap<String, PythonType>();
    Map<Symbol, PythonType> createdTypesBySymbol = new HashMap<>();
    symbols.forEach(symbol -> {
      var type = convertToType(symbol, createdTypesBySymbol);
      members.put(symbol.name(), type);
    });
    var module = new ModuleType(name, parent);
    module.members().putAll(members);

    Optional.ofNullable(parent)
      .map(ModuleType::members)
      .ifPresent(m -> m.put(name, module));

    return module;
  }

  private PythonType convertToFunctionType(FunctionSymbol symbol, Map<Symbol, PythonType> createdTypesBySymbol) {
    if (createdTypesBySymbol.containsKey(symbol)) {
      return createdTypesBySymbol.get(symbol);
    }

    var parameters = symbol.parameters()
      .stream()
      .map(SymbolsModuleTypeProvider::convertParameter)
      .toList();

    InferredType inferredType = ((FunctionSymbolImpl) symbol).declaredReturnType();
    ClassSymbol classSymbol = inferredType.runtimeTypeSymbol();
    var returnType = resolveReturnType(createdTypesBySymbol, classSymbol);

    FunctionTypeBuilder functionTypeBuilder =
      new FunctionTypeBuilder(symbol.name())
        .withAttributes(List.of())
        .withParameters(parameters)
        .withReturnType(returnType)
        .withAsynchronous(symbol.isAsynchronous())
        .withHasDecorators(symbol.hasDecorators())
        .withInstanceMethod(symbol.isInstanceMethod())
        .withHasVariadicParameter(symbol.hasVariadicParameter())
        .withDefinitionLocation(symbol.definitionLocation());
    FunctionType functionType = functionTypeBuilder.build();
    if (returnType instanceof LazyType lazyType) {
      lazyType.addConsumer(functionType::resolveLazyReturnType);
    }
    createdTypesBySymbol.put(symbol, functionType);
    return functionType;
  }

  private PythonType resolveReturnType(Map<Symbol, PythonType> createdTypesBySymbol, @Nullable ClassSymbol classSymbol) {
    if (classSymbol == null) {
      return PythonType.UNKNOWN;
    }
    if (createdTypesBySymbol.containsKey(classSymbol)) {
      return createdTypesBySymbol.get(classSymbol);
    }
    String fullyQualifiedName = classSymbol.fullyQualifiedName();
    if (fullyQualifiedName == null) {
      return convertToType(classSymbol, createdTypesBySymbol);
    }
    return resolvePossibleLazyType(createdTypesBySymbol, classSymbol, fullyQualifiedName);
  }

  PythonType resolvePossibleLazyType(Map<Symbol, PythonType> createdTypesBySymbol, ClassSymbol classSymbol, String fullyQualifiedName) {
    PythonType currentModule = rootModule;
    String[] split = fullyQualifiedName.split("\\.");
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < split.length; i++) {
      String currentElement = split[i];
      if (!sb.isEmpty()) {
        sb.append(".");
      }
      sb.append(currentElement);
      Optional<PythonType> pythonType = currentModule == null ? Optional.empty() : currentModule.resolveMember(currentElement);
      if (pythonType.isEmpty()) {
        // Either the containing module has not been resolved yet, or the type is not present
        // We need to create a LazyType for it
        String missingFqn = sb.toString();
        String[] remainingParts = Arrays.copyOfRange(split, i + 1, split.length);
        return createLazyType(fullyQualifiedName, missingFqn, remainingParts);
      }
      PythonType currentType = pythonType.get();
      if (i == split.length - 1) {
        // We have resolved the type we were looking for, return it
        return currentType;
      }
      if (!(currentType instanceof ModuleType moduleType)) {
        break;
      }
      currentModule = moduleType;
    }
    return convertToType(classSymbol, createdTypesBySymbol);
  }

  private LazyType createLazyType(String fullyQualifiedName, String prefix, String[] remainingParts) {
    Queue<String> missingModules = new ArrayDeque<>();
    String missingFqn = prefix;
    StringBuilder sb = new StringBuilder(missingFqn);
    if (!missingFqn.equals(fullyQualifiedName)) {
      // This means a module is missing, otherwise it means only the type is missing
      missingModules.add(missingFqn);
    }
    for (int j = 0; j < remainingParts.length - 1; j++) {
      sb.append(".").append(remainingParts[j]);
      missingFqn = sb.toString();
      if (!missingFqn.equals(fullyQualifiedName)) {
        missingModules.add(missingFqn);
      }
    }
    return lazyTypesContext.getOrCreateLazyType(fullyQualifiedName, missingModules, this);
  }

  private PythonType convertToClassType(ClassSymbol symbol, Map<Symbol, PythonType> createdTypesBySymbol) {
    if (createdTypesBySymbol.containsKey(symbol)) {
      return createdTypesBySymbol.get(symbol);
    }
    ClassType classType = new ClassType(symbol.name(), symbol.definitionLocation());
    createdTypesBySymbol.put(symbol, classType);
    Set<Member> members =
      symbol.declaredMembers().stream().map(m -> new Member(m.name(), convertToType(m, createdTypesBySymbol))).collect(Collectors.toSet());
    classType.members().addAll(members);


    Optional.of(symbol)
      .filter(ClassSymbolImpl.class::isInstance)
      .map(ClassSymbolImpl.class::cast)
      .filter(ClassSymbolImpl::shouldSearchHierarchyInTypeshed)
      .map(ClassSymbol::superClassesFqn)
      .map(fqns -> fqns.stream().map(this::typeshedSymbolWithFQN))
      .or(() -> Optional.of(symbol)
        .map(ClassSymbol::superClasses)
        .map(Collection::stream))
      .stream()
      .flatMap(Function.identity())
      .map(s -> convertToType(s, createdTypesBySymbol))
      .forEach(classType.superClasses()::add);

    return classType;
  }

  private static ParameterV2 convertParameter(FunctionSymbol.Parameter parameter) {
    return new ParameterV2(parameter.name(),
      PythonType.UNKNOWN,
      parameter.hasDefaultValue(),
      parameter.isKeywordOnly(),
      parameter.isPositionalOnly(),
      parameter.isKeywordVariadic(),
      parameter.isPositionalVariadic(),
      null);
  }

  private Symbol typeshedSymbolWithFQN(String fullyQualifiedName) {
    String[] fqnSplitByDot = fullyQualifiedName.split("\\.");
    String localName = fqnSplitByDot[fqnSplitByDot.length - 1];
    Symbol symbol = typeShed.symbolWithFQN(fullyQualifiedName);
    return symbol == null ? new SymbolImpl(localName, fullyQualifiedName) : ((SymbolImpl) symbol).copyWithoutUsages();
  }

  private PythonType convertToUnionType(AmbiguousSymbol ambiguousSymbol, Map<Symbol, PythonType> createdTypesBySymbol) {
    Set<PythonType> pythonTypes = ambiguousSymbol.alternatives().stream().map(a -> convertToType(a, createdTypesBySymbol)).collect(Collectors.toSet());
    return new UnionType(pythonTypes);
  }

  private PythonType convertToType(Symbol symbol, Map<Symbol, PythonType> createdTypesBySymbol) {
    return switch (symbol.kind()) {
      case CLASS -> convertToClassType((ClassSymbol) symbol, createdTypesBySymbol);
      case FUNCTION -> convertToFunctionType((FunctionSymbol) symbol, createdTypesBySymbol);
      case AMBIGUOUS -> convertToUnionType((AmbiguousSymbol) symbol, createdTypesBySymbol);
      // Symbols that are neither classes or function nor ambiguous symbols whose alternatives are all classes or functions are considered of unknown type
      case OTHER -> PythonType.UNKNOWN;
    };
  }
}
