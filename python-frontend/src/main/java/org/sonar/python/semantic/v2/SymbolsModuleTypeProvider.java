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
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.converter.AnyDescriptorToPythonTypeConverter;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeOrigin;
import org.sonar.python.types.v2.TypeWrapper;
import org.sonar.python.types.v2.UnionType;

public class SymbolsModuleTypeProvider {
  public static final String OBJECT_TYPE_FQN = "object";
  private final ProjectLevelSymbolTable projectLevelSymbolTable;
  private final ModuleType rootModule;
  private final LazyTypesContext lazyTypesContext;
  private final AnyDescriptorToPythonTypeConverter anyDescriptorToPythonTypeConverter;

  public SymbolsModuleTypeProvider(ProjectLevelSymbolTable projectLevelSymbolTable, LazyTypesContext lazyTypeContext) {
    this.projectLevelSymbolTable = projectLevelSymbolTable;
    this.lazyTypesContext = lazyTypeContext;
    this.anyDescriptorToPythonTypeConverter = new AnyDescriptorToPythonTypeConverter(lazyTypesContext);

    var rootModuleMembers = projectLevelSymbolTable.typeShedDescriptorsProvider().builtinDescriptors()
      .entrySet()
      .stream()
      .collect(Collectors.toMap(Map.Entry::getKey, e -> anyDescriptorToPythonTypeConverter.convert(e.getValue())));
    this.rootModule = new ModuleType(null, null, rootModuleMembers);
  }

  public ModuleType createBuiltinModule() {
    return rootModule;
  }

  public PythonType convertModuleType(List<String> moduleFqn, ModuleType parent) {
    var moduleName = moduleFqn.get(moduleFqn.size() - 1);
    var moduleFqnString = getModuleFqnString(moduleFqn);
    Optional<ModuleType> result =  createModuleTypeFromProjectLevelSymbolTable(moduleName, moduleFqnString, parent)
      .or(() -> createModuleTypeFromTypeShed(moduleName, moduleFqnString, parent));
    if (result.isEmpty()) {
      return PythonType.UNKNOWN;
    }
    return result.get();
  }

  private static String getModuleFqnString(List<String> moduleFqn) {
    return String.join(".", moduleFqn);
  }

  private Optional<ModuleType> createModuleTypeFromProjectLevelSymbolTable(String moduleName, String moduleFqn, ModuleType parent) {
    return Optional.ofNullable(projectLevelSymbolTable.getSymbolsFromModule(moduleFqn))
      .map(projectModuleSymbols -> createModuleFromSymbols(moduleName, parent, projectModuleSymbols));
  }

  private Optional<ModuleType> createModuleTypeFromTypeShed(String moduleName, String moduleFqn, ModuleType parent) {
    var moduleMembers = projectLevelSymbolTable.typeShedDescriptorsProvider().descriptorsForModule(moduleFqn)
      .entrySet().stream()
      .collect(Collectors.toMap(Map.Entry::getKey, e -> anyDescriptorToPythonTypeConverter.convert(e.getValue())));
    return Optional.of(moduleMembers).filter(m -> !m.isEmpty())
      .map(m -> new ModuleType(moduleName, parent, m));
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
      .map(this::convertParameter)
      .toList();

    var returnType = PythonType.UNKNOWN;

    TypeOrigin typeOrigin = symbol.isStub() ? TypeOrigin.STUB : TypeOrigin.LOCAL;

    FunctionTypeBuilder functionTypeBuilder =
      new FunctionTypeBuilder(symbol.name())
        .withAttributes(List.of())
        .withParameters(parameters)
        .withReturnType(returnType)
        .withTypeOrigin(typeOrigin)
        .withAsynchronous(symbol.isAsynchronous())
        .withHasDecorators(symbol.hasDecorators())
        .withInstanceMethod(symbol.isInstanceMethod())
        .withHasVariadicParameter(symbol.hasVariadicParameter())
        .withDefinitionLocation(symbol.definitionLocation());
    FunctionType functionType = functionTypeBuilder.build();
    createdTypesBySymbol.put(symbol, functionType);
    return functionType;
  }

  PythonType resolvePossibleLazyType(String fullyQualifiedName) {
    if (rootModule == null) {
      // If root module has not been created yet, return lazy type
      return lazyTypesContext.getOrCreateLazyType(fullyQualifiedName);
    }
    PythonType currentType = rootModule;
    String[] fqnParts = fullyQualifiedName.split("\\.");
    var fqnPartsQueue = new ArrayDeque<>(Arrays.asList(fqnParts));
    while (!fqnPartsQueue.isEmpty()) {
      var memberName = fqnPartsQueue.poll();
      var memberOpt = currentType.resolveMember(memberName);
      if (memberOpt.isEmpty()) {
        if (currentType instanceof ModuleType) {
          // The type is part of an unresolved submodule
          // Create a lazy type for it
          return lazyTypesContext.getOrCreateLazyType(fullyQualifiedName);
        } else {
          // The type is an unknown member of an already resolved type
          // Default to UNKNOWN
          return PythonType.UNKNOWN;
        }
      }
      currentType = memberOpt.get();
    }
    return currentType;
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
      .map(fqns -> fqns.stream().map(this::resolvePossibleLazyType))
      .or(() -> Optional.of(symbol)
        .map(ClassSymbol::superClasses)
        .map(Collection::stream)
        .map(symbols -> symbols.map(s -> convertToType(s, createdTypesBySymbol))))
      .stream()
      .flatMap(Function.identity())
      .map(LazyTypeWrapper::new)
      .forEach(classType.superClasses()::add);

    return classType;
  }

  private ParameterV2 convertParameter(FunctionSymbol.Parameter parameter) {
    var typeWrapper = Optional.ofNullable(((FunctionSymbolImpl.ParameterImpl) parameter).annotatedTypeName())
      .map(lazyTypesContext::getOrCreateLazyTypeWrapper)
      .orElse(TypeWrapper.UNKNOWN_TYPE_WRAPPER);

    return new ParameterV2(parameter.name(),
      typeWrapper,
      parameter.hasDefaultValue(),
      parameter.isKeywordOnly(),
      parameter.isPositionalOnly(),
      parameter.isKeywordVariadic(),
      parameter.isPositionalVariadic(),
      parameter.location());
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
