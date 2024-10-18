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
import java.util.stream.IntStream;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnknownType;

public class ProjectLevelTypeTable {

  private final SymbolsModuleTypeProvider symbolsModuleTypeProvider;
  private final ModuleType rootModule;
  private final LazyTypesContext lazyTypesContext;
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

  public ProjectLevelTypeTable(ProjectLevelSymbolTable projectLevelSymbolTable) {
    this.lazyTypesContext = new LazyTypesContext(this);
    this.symbolsModuleTypeProvider = new SymbolsModuleTypeProvider(projectLevelSymbolTable, lazyTypesContext);
    this.rootModule = this.symbolsModuleTypeProvider.createBuiltinModule();
  }

  public ModuleType getBuiltinsModule() {
    return rootModule;
  }

  public PythonType getType(String typeFqn) {
    return getType(typeFqn.split("\\."));
  }

  public PythonType getType(String... typeFqnParts) {
    return getType(List.of(typeFqnParts));
  }

  public PythonType getType(List<String> typeFqnParts) {
    var parent = (PythonType) rootModule;
    for (int i = 0; i < typeFqnParts.size(); i++) {
      var part = typeFqnParts.get(i);
      if (parent instanceof ObjectType) {
        return PythonType.UNKNOWN;
      }
      Optional<PythonType> resolvedMember = parent.resolveMember(part);
      if (resolvedMember.isPresent()) {
        parent = resolvedMember.get();
      } else if (parent instanceof ModuleType module) {
        var moduleFqnParts = IntStream.rangeClosed(0, i)
          .mapToObj(typeFqnParts::get)
          .toList();
        parent = symbolsModuleTypeProvider.convertModuleType(moduleFqnParts, module);
        if (parent instanceof ModuleType moduleType) {
          addAliasMembers(moduleFqnParts, moduleType);
        }
      } else {
        return PythonType.UNKNOWN;
      }
    }
    return parent;
  }

  public LazyTypesContext lazyTypesContext() {
    return lazyTypesContext;
  }

  public void addAliasMembers(List<String> moduleFqnParts, ModuleType moduleType) {
    String moduleFqn = String.join(".", moduleFqnParts);
    aliasMembers.getOrDefault(moduleFqn, Map.of())
      .forEach((memberName, alias) -> {
        var pythonType = getType(alias);
        if (!(pythonType instanceof UnknownType)) {
          moduleType.members().put(memberName, pythonType);
        }
      });
  }
}
