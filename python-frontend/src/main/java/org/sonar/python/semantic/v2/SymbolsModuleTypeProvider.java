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

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

public class SymbolsModuleTypeProvider {
  private final ProjectLevelSymbolTable projectLevelSymbolTable;

  public SymbolsModuleTypeProvider(ProjectLevelSymbolTable projectLevelSymbolTable) {
    this.projectLevelSymbolTable = projectLevelSymbolTable;
  }

  public ModuleType createBuiltinModule() {
    return createModuleFromSymbols(null, null, TypeShed.builtinSymbols().values());
  }

  public ModuleType createModuleType(List<String> moduleFqn, ModuleType parent) {
    var moduleName = moduleFqn.get(moduleFqn.size() - 1);
    var moduleFqnString = getModuleFqnString(moduleFqn);
    return createModuleTypeFromProjectLevelSymbolTable(moduleName, moduleFqnString, parent)
      .or(() -> createModuleTypeFromTypeShed(moduleName, moduleFqnString, parent))
      .orElseGet(() -> createEmptyModule(moduleName, parent));
  }

  private static String getModuleFqnString(List<String> moduleFqn) {
    return String.join(".", moduleFqn);
  }

  private Optional<ModuleType> createModuleTypeFromProjectLevelSymbolTable(String moduleName, String moduleFqn, ModuleType parent) {
    return Optional.ofNullable(projectLevelSymbolTable.getSymbolsFromModule(moduleFqn))
      .map(projectModuleSymbols -> createModuleFromSymbols(moduleName, parent, projectModuleSymbols));
  }

  private Optional<ModuleType> createModuleTypeFromTypeShed(String moduleName, String moduleFqn, ModuleType parent) {
    return Optional.ofNullable(TypeShed.symbolsForModule(moduleFqn))
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
    symbols.forEach(symbol -> {
      var type = convertToType(symbol, new HashMap<>());
      members.put(symbol.name(), type);
    });
    var module = new ModuleType(name, parent);
    module.members().putAll(members);

    Optional.ofNullable(parent)
      .map(ModuleType::members)
      .ifPresent(m -> m.put(name, module));

    return module;
  }

  private static PythonType convertToObjectType(Symbol symbol) {
    // What should we have here?
    return PythonType.UNKNOWN;
  }

  private static PythonType convertToFunctionType(FunctionSymbol symbol, Map<Symbol, PythonType> createdTypesBySymbol) {
    if (createdTypesBySymbol.containsKey(symbol)) {
      return createdTypesBySymbol.get(symbol);
    }

    var parameters = symbol.parameters()
      .stream()
      .map(SymbolsModuleTypeProvider::convertParameter)
      .toList();

    FunctionTypeBuilder functionTypeBuilder =
      new FunctionTypeBuilder(symbol.name())
        .withAttributes(List.of())
        .withParameters(parameters)
        .withReturnType(PythonType.UNKNOWN)
        .withAsynchronous(symbol.isAsynchronous())
        .withHasDecorators(symbol.hasDecorators())
        .withInstanceMethod(symbol.isInstanceMethod())
        .withHasVariadicParameter(symbol.hasVariadicParameter())
        .withDefinitionLocation(symbol.definitionLocation());
    FunctionType functionType = functionTypeBuilder.build();
    createdTypesBySymbol.put(symbol, functionType);
    return functionType;
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

  private PythonType convertToClassType(ClassSymbol symbol, Map<Symbol, PythonType> createdTypesBySymbol) {
    if (createdTypesBySymbol.containsKey(symbol)) {
      return createdTypesBySymbol.get(symbol);
    }
    ClassType classType = new ClassType(symbol.name(), symbol.definitionLocation());
    createdTypesBySymbol.put(symbol, classType);
    Set<Member> members =
      symbol.declaredMembers().stream().map(m -> new Member(m.name(), convertToType(m, createdTypesBySymbol))).collect(Collectors.toSet());
    classType.members().addAll(members);
    List<PythonType> superClasses = symbol.superClasses().stream().map(s -> convertToType(s, createdTypesBySymbol)).toList();
    classType.superClasses().addAll(superClasses);
    return classType;
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
      case OTHER -> convertToObjectType(symbol);
    };
  }

}
