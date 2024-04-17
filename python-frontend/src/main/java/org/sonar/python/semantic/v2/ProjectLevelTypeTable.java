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

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
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
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

public class ProjectLevelTypeTable {

  private final ProjectLevelSymbolTable projectLevelSymbolTable;
  private final Map<String, PythonType> builtinTypes;
  private final Map<String, ModuleType> modules;

  public ProjectLevelTypeTable(ProjectLevelSymbolTable projectLevelSymbolTable) {
    this.projectLevelSymbolTable = projectLevelSymbolTable;
    this.modules = new HashMap<>();
    Map<String, Symbol> builtinSymbols = TypeShed.builtinSymbols();
    this.builtinTypes = new HashMap<>();
    builtinSymbols.forEach((key, value) -> builtinTypes.put(key, convertToType(value)));
    modules.put("builtins", new ModuleType("builtins", builtinTypes));
  }

  public ModuleType getModule(String moduleName) {
    if (modules.containsKey(moduleName)) {
      return modules.get(moduleName);
    }
    Set<Symbol> symbolsFromModule = projectLevelSymbolTable.getSymbolsFromModule(moduleName);
    Map<String, PythonType> children;
    if (symbolsFromModule != null) {
      children = symbolsFromModule.stream()
        .map(this::convertToType)
        .collect(Collectors.toMap(PythonType::name, Function.identity(), (a, b) -> PythonType.UNKNOWN));
    } else {
      // FIXME: what to do here?
      children = Collections.emptyMap();
    }

    ModuleType moduleType = new ModuleType(moduleName, children);
    modules.put(moduleName, moduleType);
    return moduleType;
  }

  private PythonType convertToType(Symbol symbol) {
    return switch (symbol.kind()) {
      case CLASS -> converToClassType((ClassSymbol) symbol);
      case FUNCTION -> convertToFunctionType((FunctionSymbol) symbol);
      case AMBIGUOUS -> convertToUnionType((AmbiguousSymbol) symbol);
      case OTHER -> converToObjectType(symbol);
    };
  }

  private PythonType converToObjectType(Symbol symbol) {
    // What should we have here?
    return PythonType.UNKNOWN;
  }

  private PythonType convertToFunctionType(FunctionSymbol symbol) {
    return new FunctionType(symbol.name(), List.of(), List.of(), PythonType.UNKNOWN, false, false, false, false, null);
  }

  private PythonType converToClassType(ClassSymbol symbol) {
    Set<Member> members = symbol.declaredMembers().stream().map(m -> new Member(m.name(), convertToType(m))).collect(Collectors.toSet());
    List<PythonType> superClasses = symbol.superClasses().stream().map(this::convertToType).toList();
    return new ClassType(symbol.name(), members, List.of(), superClasses);
  }

  private PythonType convertToUnionType(AmbiguousSymbol ambiguousSymbol) {
    List<PythonType> pythonTypes = ambiguousSymbol.alternatives().stream().map(this::convertToType).toList();
    return new UnionType(pythonTypes);
  }
}
