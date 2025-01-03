/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.semantic.v2;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.converter.AnyDescriptorToPythonTypeConverter;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeOrigin;
import org.sonar.python.types.v2.TypeWrapper;

public class SymbolsModuleTypeProvider {
  private final ProjectLevelSymbolTable projectLevelSymbolTable;
  private final ModuleType rootModule;
  private final LazyTypesContext lazyTypesContext;
  private final AnyDescriptorToPythonTypeConverter anyDescriptorToPythonTypeConverter;
  private final Map<String, Map<String, String>> aliasMembers = Map.ofEntries(
    Map.entry("typing", Map.ofEntries(
      Map.entry("List", "list"),
      Map.entry("Tuple", "tuple"),
      Map.entry("Dict", "dict"),
      Map.entry("Set", "set"),
      Map.entry("FrozenSet", "frozenset"),
      Map.entry("Type", "type")
    ))
  );

  public SymbolsModuleTypeProvider(ProjectLevelSymbolTable projectLevelSymbolTable, LazyTypesContext lazyTypeContext) {
    this.projectLevelSymbolTable = projectLevelSymbolTable;
    this.lazyTypesContext = lazyTypeContext;
    this.anyDescriptorToPythonTypeConverter = new AnyDescriptorToPythonTypeConverter(lazyTypesContext);

    var rootModuleMembers = projectLevelSymbolTable.typeShedDescriptorsProvider().builtinDescriptors()
      .entrySet()
      .stream()
      .collect(Collectors.toMap(Map.Entry::getKey, e -> TypeWrapper.of(anyDescriptorToPythonTypeConverter.convert("", e.getValue(), TypeOrigin.STUB))));
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
    var retrieved = projectLevelSymbolTable.getDescriptorsFromModule(moduleFqn);
    if (retrieved == null) {
      return Optional.empty();
    }
    var members = retrieved.stream()
      .collect(Collectors.toMap(Descriptor::name,
        d -> TypeWrapper.of(anyDescriptorToPythonTypeConverter.convert(moduleFqn, d, TypeOrigin.LOCAL))
      ));
    return Optional.of(createModuleType(moduleName, moduleFqn, parent, members));
  }

  private Optional<ModuleType> createModuleTypeFromTypeShed(String moduleName, String moduleFqn, ModuleType parent) {
    Map<String, Descriptor> stringDescriptorMap = projectLevelSymbolTable.typeShedDescriptorsProvider().descriptorsForModule(moduleFqn);
    Map<String, TypeWrapper> members = anyDescriptorToPythonTypeConverter.convertModuleType(moduleFqn, stringDescriptorMap);
    return Optional.of(members).filter(m -> !m.isEmpty()).map(m -> createModuleType(moduleName, moduleFqn, parent, m));
  }

  private ModuleType createModuleType(String moduleName, String moduleFqn, ModuleType parent, Map<String, TypeWrapper> members) {
    addTypingAliases(moduleFqn, members);
    return new ModuleType(moduleName, parent, members);
  }

  private void addTypingAliases(String moduleFqn, Map<String, TypeWrapper> members) {
    aliasMembers.getOrDefault(moduleFqn, Map.of()).forEach((alias, original) -> {
      var originalType = rootModule.members().get(original);
      if (originalType != null) {
        members.put(alias, originalType);
      }
    });
  }
}
