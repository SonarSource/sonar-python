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
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.protobuf.SymbolsProtos;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.SimpleTypeWrapper;
import org.sonar.python.types.v2.TypeOrigin;
import org.sonar.python.types.v2.TypeWrapper;
import org.sonar.python.types.v2.UnionType;

public class SymbolsModuleTypeProvider {
  public static final String OBJECT_TYPE_FQN = "object";
  private final ProjectLevelSymbolTable projectLevelSymbolTable;
  private final TypeShed typeShed;
  private ModuleType rootModule;
  private LazyTypesContext lazyTypesContext;

  public SymbolsModuleTypeProvider(ProjectLevelSymbolTable projectLevelSymbolTable, TypeShed typeShed, LazyTypesContext lazyTypeContext) {
    this.projectLevelSymbolTable = projectLevelSymbolTable;
    this.typeShed = typeShed;
    this.lazyTypesContext = lazyTypeContext;
    this.rootModule = createModuleFromSymbols(null, null, typeShed.builtinSymbols().values());
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
    return Optional.ofNullable(typeShed.symbolsForModule(moduleFqn))
      .filter(Predicate.not(Map::isEmpty))
      .map(typeShedModuleSymbols -> createModuleFromSymbols(moduleName, parent, typeShedModuleSymbols.values()));
  }

  private ModuleType createModuleFromSymbols(@Nullable String name, @Nullable ModuleType parent, Collection<Symbol> symbols) {
    var members = new HashMap<String, PythonType>();
    Map<Symbol, PythonType> createdTypesBySymbol = new HashMap<>();
    var module = new ModuleType(name, parent);
    symbols.forEach(symbol -> {
      var type = convertToType(symbol, createdTypesBySymbol, module);
      members.put(symbol.name(), type);
    });
    module.members().putAll(members);

    Optional.ofNullable(parent)
      .map(ModuleType::members)
      .ifPresent(m -> m.put(name, module));

    return module;
  }

  private PythonType convertToFunctionType(FunctionSymbol symbol, Map<Symbol, PythonType> createdTypesBySymbol, ModuleType parent) {
    if (createdTypesBySymbol.containsKey(symbol)) {
      return createdTypesBySymbol.get(symbol);
    }

    var parameters = symbol.parameters()
      .stream()
      .map(this::convertParameter)
      .toList();

    var returnType = getReturnTypeFromSymbol(symbol);

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
        .withDefinitionLocation(symbol.definitionLocation())
        .withOwner(parent);
    FunctionType functionType = functionTypeBuilder.build();
    createdTypesBySymbol.put(symbol, functionType);
    return functionType;
  }

  private TypeWrapper getReturnTypeFromSymbol(FunctionSymbol symbol) {
    var returnTypeFqns = getReturnTypeFqn(symbol);
    var returnTypeList = returnTypeFqns.stream().map(lazyTypesContext::getOrCreateLazyTypeWrapper).map(ObjectType::new).toList();
    //TODO Support type unions (SONARPY-2132)
    var type = returnTypeList.size() == 1 ? returnTypeList.get(0) : PythonType.UNKNOWN;
    return new SimpleTypeWrapper(type);
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

  private PythonType convertToClassType(ClassSymbol symbol, Map<Symbol, PythonType> createdTypesBySymbol, ModuleType parent) {
    if (createdTypesBySymbol.containsKey(symbol)) {
      return createdTypesBySymbol.get(symbol);
    }
    ClassType classType = new ClassType(symbol.name(), symbol.definitionLocation());
    createdTypesBySymbol.put(symbol, classType);
    Set<Member> members =
      symbol.declaredMembers().stream().map(m -> new Member(m.name(), convertToType(m, createdTypesBySymbol, parent))).collect(Collectors.toSet());
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
        .map(symbols -> symbols.map(s -> convertToType(s, createdTypesBySymbol, parent))))
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

  private PythonType convertToUnionType(AmbiguousSymbol ambiguousSymbol, Map<Symbol, PythonType> createdTypesBySymbol, ModuleType parent) {
    Set<PythonType> pythonTypes = ambiguousSymbol.alternatives().stream().map(a -> convertToType(a, createdTypesBySymbol, parent)).collect(Collectors.toSet());
    return new UnionType(pythonTypes);
  }

  private PythonType convertToType(Symbol symbol, Map<Symbol, PythonType> createdTypesBySymbol, ModuleType parent) {
    return switch (symbol.kind()) {
      case CLASS -> convertToClassType((ClassSymbol) symbol, createdTypesBySymbol, parent);
      case FUNCTION -> convertToFunctionType((FunctionSymbol) symbol, createdTypesBySymbol, parent);
      case AMBIGUOUS -> convertToUnionType((AmbiguousSymbol) symbol, createdTypesBySymbol, parent);
      // Symbols that are neither classes or function nor ambiguous symbols whose alternatives are all classes or functions are considered of unknown type
      case OTHER -> PythonType.UNKNOWN;
    };
  }

  private List<String> getReturnTypeFqn(FunctionSymbol symbol) {
    List<String> fqnList = List.of();
    if (symbol.annotatedReturnTypeName() != null && !OBJECT_TYPE_FQN.equals(symbol.annotatedReturnTypeName())) {
      fqnList = List.of(symbol.annotatedReturnTypeName());
    } else if (symbol instanceof FunctionSymbolImpl functionSymbol && functionSymbol.protobufReturnType() != null) {
      var protoReturnType = functionSymbol.protobufReturnType();
      fqnList = getSymbolTypeFqn(protoReturnType);
    }
    return fqnList;
  }

  List<String> getSymbolTypeFqn(SymbolsProtos.Type type) {
    if (OBJECT_TYPE_FQN.equals(type.getFullyQualifiedName())) {
      return List.of();
    }
    switch (type.getKind()) {
      case INSTANCE:
        String typeName = type.getFullyQualifiedName();
        // _SpecialForm is the type used for some special types, like Callable, Union, TypeVar, ...
        // It comes from CPython impl: https://github.com/python/cpython/blob/e39ae6bef2c357a88e232dcab2e4b4c0f367544b/Lib/typing.py#L439
        // This doesn't seem to be very precisely specified in typeshed, because it has special semantic.
        // To avoid FPs, we treat it as ANY
        if ("typing._SpecialForm".equals(typeName)) {
          return List.of();
        }
        typeName = typeName.replaceFirst("^builtins\\.", "");
        return typeName.isEmpty() ? List.of() : List.of(typeName);
      case TYPE:
        return List.of("type");
      case TYPE_ALIAS:
        return getSymbolTypeFqn(type.getArgs(0));
      case CALLABLE:
        // this should be handled as a function type - see SONARPY-953
        // Creates FPs with `sys.gettrace`
        return List.of();
      case UNION:
        return type.getArgsList().stream().map(this::getSymbolTypeFqn).flatMap(Collection::stream).toList();
      case TUPLE:
        return List.of("tuple");
      case NONE:
        return List.of("NoneType");
      case TYPED_DICT:
        return List.of("dict");
      case TYPE_VAR:
        return Optional.of(type)
          .filter(InferredTypes::filterTypeVar)
          .map(SymbolsProtos.Type::getFullyQualifiedName)
          .map(List::of)
          .orElseGet(List::of);
      default:
        return List.of();
    }
  }

}
