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
package org.sonar.python.semantic.v2.typeshed;

import com.google.protobuf.ProtocolStringList;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.types.protobuf.SymbolsProtos;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

public class TypeShedModuleTypeProvider {
  private static final Map<String, String> MODULES_TO_DISAMBIGUATE = Map.of(
    "ConfigParser", "2@ConfigParser",
    "Queue", "2@Queue",
    "SocketServer", "2@SocketServer"
  );
  private static final String PROTOBUF_BASE_RESOURCE_PATH = "/org/sonar/python/types/";
  private static final String PROTOBUF_CUSTOM_STUBS = PROTOBUF_BASE_RESOURCE_PATH + "custom_protobuf/";
  private static final String PROTOBUF = PROTOBUF_BASE_RESOURCE_PATH + "stdlib_protobuf/";
  private static final String PROTOBUF_THIRD_PARTY = PROTOBUF_BASE_RESOURCE_PATH + "third_party_protobuf/";
  private static final String PROTOBUF_THIRD_PARTY_MYPY = PROTOBUF_BASE_RESOURCE_PATH + "third_party_protobuf_mypy/";
  private static final String BUILTINS_FQN = "builtins";
  private static final Map<String, Consumer<ModuleType>> MODULE_TYPE_ENRICHERS = Map.of(
    BUILTINS_FQN, TypeShedModuleTypeProvider::enrichBuiltinsModule
  );

  private Set<String> supportedPythonVersions;
  private final Map<String, ModuleType> modulesByFqn;

  public TypeShedModuleTypeProvider() {
    this.modulesByFqn = new HashMap<>();
  }

  private static void enrichBuiltinsModule(ModuleType builtins) {
    builtins.members().put(NONE_TYPE, new ClassType(NONE_TYPE));
  }

  public ModuleType getBuiltinModuleType() {
    supportedPythonVersions = ProjectPythonVersion.currentVersions().stream().map(PythonVersionUtils.Version::serializedValue).collect(Collectors.toSet());
    return getModuleType(BUILTINS_FQN, null);
  }

  @CheckForNull
  public ModuleType getModuleType(String moduleName, @Nullable ModuleType parent) {
    if (modulesByFqn.containsKey(moduleName)) {
      return modulesByFqn.get(moduleName);
    }

    var context = new ReadModulesContext(modulesByFqn)
      .addModuleToRead(moduleName);

    while (context.hasModulesToRead()) {
      var name = context.nextModuleToRead();

      var type = Stream.of(PROTOBUF_CUSTOM_STUBS, PROTOBUF, PROTOBUF_THIRD_PARTY_MYPY, PROTOBUF_THIRD_PARTY)
        .map(dirName -> getModuleTypeFromProtobufModule(context, name, dirName))
        .filter(Objects::nonNull)
        .findFirst()
        .orElse(null);

      if (MODULE_TYPE_ENRICHERS.containsKey(name)) {
        MODULE_TYPE_ENRICHERS.get(name).accept(type);
      }
      context.addResolvedModule(name, type);
    }

    resolvePromiseTypes(context);
    modulesByFqn.putAll(context.resolvedModules());
    if (context.resolvedModules().containsKey(moduleName)) {
      var moduleType = context.resolvedModules().get(moduleName);
      moduleType = new ModuleType(moduleType.name(), parent, moduleType.members());
      modulesByFqn.put(moduleName, moduleType);
      return moduleType;
    } else {
      return null;
    }
  }

  private void resolvePromiseTypes(ReadModulesContext context) {
    context.promiseTypes().forEach((fqn, promiseType) -> resolvePromiseType(context, fqn, promiseType));
  }

  private void resolvePromiseType(ReadModulesContext context, String typeFqn, PromiseType promiseType) {
    var moduleFqn = getModuleName(typeFqn);
    var typeName = getName(typeFqn);
    var resolvedType = context.resolvedModules().containsKey(moduleFqn) ?
      context.resolvedModules().get(moduleFqn).resolveMember(typeName).orElse(PythonType.UNKNOWN)
      : PythonType.UNKNOWN;
    promiseType.resolve(resolvedType);
  }

  @CheckForNull
  private ModuleType getModuleTypeFromProtobufModule(ReadModulesContext context, String moduleName, String dirName) {
    return Optional.ofNullable(readModuleSymbol(moduleName, dirName))
      .map(moduleSymbol -> fromModuleSymbol(context, moduleSymbol))
      .orElse(null);
  }

  @CheckForNull
  private SymbolsProtos.ModuleSymbol readModuleSymbol(String moduleName, String dirName) {
    var fileName = MODULES_TO_DISAMBIGUATE.getOrDefault(moduleName, moduleName);
    try (var resource = this.getClass().getResourceAsStream(dirName + fileName + ".protobuf")) {
      if (resource == null) {
        return null;
      }
      return SymbolsProtos.ModuleSymbol.parseFrom(resource);
    } catch (IOException e) {
      return null;
    }
  }

  private ModuleType fromModuleSymbol(ReadModulesContext context, SymbolsProtos.ModuleSymbol moduleSymbol) {
    var moduleType = new ModuleType(getName(moduleSymbol.getFullyQualifiedName()), null);

    var classes = fromSymbols(moduleSymbol.getClassesList(),
      SymbolsProtos.ClassSymbol::getName,
      SymbolsProtos.ClassSymbol::getValidForList,
      s -> fromClassSymbol(context, s));

    var functions = fromSymbols(moduleSymbol.getFunctionsList(),
      SymbolsProtos.FunctionSymbol::getName,
      SymbolsProtos.FunctionSymbol::getValidForList,
      s -> fromFunctionSymbol(context, s, moduleType));

    var overloadedFunctions = fromSymbols(moduleSymbol.getOverloadedFunctionsList(),
      SymbolsProtos.OverloadedFunctionSymbol::getName,
      SymbolsProtos.OverloadedFunctionSymbol::getValidForList,
      s -> fromOverloadedFunctionSymbol(context, s, moduleType));

    var vars = fromSymbols(moduleSymbol.getVarsList(),
      SymbolsProtos.VarSymbol::getName,
      SymbolsProtos.VarSymbol::getValidForList,
      s -> fromVarSymbol(context, s));

    Stream.of(classes, functions, overloadedFunctions, vars)
      .flatMap(Function.identity())
      .forEach(entry -> moduleType.members().put(entry.getKey(), entry.getValue()));
    return moduleType;
  }

  private <T> Stream<Map.Entry<String, PythonType>> fromSymbols(
    List<T> symbols,
    Function<T, String> symbolNameProvider,
    Function<T, ProtocolStringList> symbolPythonVersionsProvider,
    Function<T, PythonType> symbolToTypeConverter) {
    return symbols.stream()
      .filter(d -> isValidForProjectPythonVersion(symbolPythonVersionsProvider.apply(d)))
      .map(s -> Map.entry(symbolNameProvider.apply(s), symbolToTypeConverter.apply(s)));
  }

  private static PythonType fromVarSymbol(ReadModulesContext context, SymbolsProtos.VarSymbol symbol) {
    return PythonType.UNKNOWN;
  }

  private PythonType fromOverloadedFunctionSymbol(ReadModulesContext context, SymbolsProtos.OverloadedFunctionSymbol symbol, PythonType owner) {
    if (symbol.getDefinitionsList().size() < 2) {
      throw new IllegalStateException("Overloaded function symbols should have at least two definitions.");
    }

    var functionTypes = symbol.getDefinitionsList().stream()
      .map(functionSymbol -> fromFunctionSymbol(context, functionSymbol, owner))
      .toList();

    return UnionType.or(functionTypes);
  }

  private PythonType fromFunctionSymbol(ReadModulesContext context, SymbolsProtos.FunctionSymbol symbol, PythonType owner) {
    var isInstanceMethod = owner instanceof ClassType && !symbol.getIsStatic() && !symbol.getIsClassMethod();
    var isAsynchronous = symbol.getIsAsynchronous();
    var hasDecorators = symbol.getHasDecorators();
    var decorators = symbol.getResolvedDecoratorNamesList();
    var returnAnnotation = symbol.getReturnAnnotation();
    var returnTypeName = returnAnnotation.getFullyQualifiedName();
    var annotatedReturnTypeName = returnTypeName.isEmpty() ? null : org.sonar.python.types.TypeShed.normalizedFqn(returnTypeName);
    var protobufReturnType = returnAnnotation;
    var hasVariadicParameter = false;
    var parameters = new ArrayList<ParameterV2>();
    for (var parameterSymbol : symbol.getParametersList()) {
      var isKeywordOnly = parameterSymbol.getKind() == SymbolsProtos.ParameterKind.KEYWORD_ONLY;
      var isPositionalOnly = parameterSymbol.getKind() == SymbolsProtos.ParameterKind.POSITIONAL_ONLY;
      boolean isPositionalVariadic = parameterSymbol.getKind() == SymbolsProtos.ParameterKind.VAR_POSITIONAL;
      boolean isKeywordVariadic = parameterSymbol.getKind() == SymbolsProtos.ParameterKind.VAR_KEYWORD;
      hasVariadicParameter |= isKeywordVariadic || isPositionalVariadic;
      var parameterName = parameterSymbol.hasName() ? parameterSymbol.getName() : null;
      // TODO: fix location
      var parameter = new ParameterV2(parameterName, PythonType.UNKNOWN, parameterSymbol.getHasDefault(), isKeywordOnly, isPositionalOnly, isKeywordVariadic, isPositionalVariadic, null);
      if (parameterSymbol.getTypeAnnotation().getKind() == SymbolsProtos.TypeKind.INSTANCE) {
        var typeAnnotation = parameterSymbol.getTypeAnnotation();
        var typeAnnotationFqn = typeAnnotation.getFullyQualifiedName();
        var parameterPromiseType = context.getOrCreatePromiseType(typeAnnotationFqn);
        parameterPromiseType.addConsumer(parameter::declaredType);
      }
      parameters.add(parameter);
    }
    // TODO: fix return type
    return new FunctionTypeBuilder(symbol.getName())
      .withOwner(owner)
      .withInstanceMethod(isInstanceMethod)
      .withParameters(parameters)
      .withAsynchronous(isAsynchronous)
      .withHasDecorators(hasDecorators)
      .withHasVariadicParameter(hasVariadicParameter)
      .withReturnType(PythonType.UNKNOWN)
      .withAttributes(List.of())
      .build();
  }

  private ClassType fromClassSymbol(ReadModulesContext context, SymbolsProtos.ClassSymbol classSymbol) {
    var classType = new ClassTypeBuilder().withName(classSymbol.getName()).build();

    classSymbol.getSuperClassesList()
      .stream()
      .forEach(superClassFqn -> {
        var promiseType = context.getOrCreatePromiseType(superClassFqn);
        promiseType.addConsumer(PromiseType.collectionTypeResolver(promiseType, classType.superClasses()));
        classType.superClasses().add(promiseType);
        context.addModuleToRead(getModuleName(superClassFqn));
      });

    fromSymbols(classSymbol.getMethodsList(),
      SymbolsProtos.FunctionSymbol::getName,
      SymbolsProtos.FunctionSymbol::getValidForList,
      s -> fromFunctionSymbol(context, s, classType))
      .map(t -> new Member(t.getValue().name(), t.getValue()))
      .forEach(classType.members()::add);

    fromSymbols(classSymbol.getOverloadedMethodsList(),
      SymbolsProtos.OverloadedFunctionSymbol::getName,
      SymbolsProtos.OverloadedFunctionSymbol::getValidForList,
      s -> fromOverloadedFunctionSymbol(context, s, classType))
      .map(t -> new Member(t.getValue().name(), t.getValue()))
      .forEach(classType.members()::add);

    return classType;
  }

  private String getModuleName(String fqn) {
    int lastIndex = fqn.lastIndexOf('.');
    if (lastIndex == -1) {
      return fqn;  // Return the original string if there's no dot
    }
    return fqn.substring(0, lastIndex);
  }

  private String getName(String fqn) {
    int lastIndex = fqn.lastIndexOf('.');
    if (lastIndex == -1) {
      return fqn;  // Return the original string if there's no dot
    }
    return fqn.substring(lastIndex + 1);
  }

  private boolean isValidForProjectPythonVersion(List<String> validForPythonVersions) {
    if (validForPythonVersions.isEmpty()) {
      return true;
    }
    if (supportedPythonVersions().stream().allMatch(PythonVersionUtils.Version.V_312.serializedValue()::equals)
      && validForPythonVersions.contains(PythonVersionUtils.Version.V_311.serializedValue())) {
      return true;
    }
    HashSet<String> intersection = new HashSet<>(validForPythonVersions);
    intersection.retainAll(supportedPythonVersions());
    return !intersection.isEmpty();
  }

  private Set<String> supportedPythonVersions() {
    if (supportedPythonVersions == null) {
      supportedPythonVersions = ProjectPythonVersion.currentVersions()
        .stream()
        .map(PythonVersionUtils.Version::serializedValue)
        .collect(Collectors.toSet());
    }
    return supportedPythonVersions;
  }


}
