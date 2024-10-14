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

public class SymbolsModuleTypeProvider {
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
      .collect(Collectors.toMap(Map.Entry::getKey, e -> anyDescriptorToPythonTypeConverter.convert(e.getValue(), TypeOrigin.STUB)));
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
    var members = retrieved.stream().collect(Collectors.toMap(Descriptor::name, d -> anyDescriptorToPythonTypeConverter.convert(d, TypeOrigin.LOCAL)));
    return Optional.of(new ModuleType(moduleName, parent, members));
  }

  private Optional<ModuleType> createModuleTypeFromTypeShed(String moduleName, String moduleFqn, ModuleType parent) {
    var moduleMembers = projectLevelSymbolTable.typeShedDescriptorsProvider().descriptorsForModule(moduleFqn)
      .entrySet().stream()
      .collect(Collectors.toMap(Map.Entry::getKey, e -> anyDescriptorToPythonTypeConverter.convert(e.getValue(), TypeOrigin.STUB)));
    return Optional.of(moduleMembers).filter(m -> !m.isEmpty())
      .map(m -> new ModuleType(moduleName, parent, m));
  }

}
