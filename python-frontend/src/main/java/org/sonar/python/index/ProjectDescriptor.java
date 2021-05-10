/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.index;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.semantic.SymbolTableBuilder;
import org.sonar.python.semantic.SymbolUtils;

public class ProjectDescriptor {
  private final Map<String, ModuleDescriptor> modules = new HashMap<>();

  public void addModule(FileInput fileInput, String packageName, PythonFile pythonFile) {
    SymbolTableBuilder symbolTableBuilder = new SymbolTableBuilder(packageName, pythonFile);
    String fullyQualifiedModuleName = SymbolUtils.fullyQualifiedModuleName(packageName, pythonFile.fileName());
    fileInput.accept(symbolTableBuilder);

    List<Descriptor> descriptors = fileInput.globalVariables().stream()
      // TODO: We don't put builtin or imported names in global symbol table to avoid duplicate FQNs in project level symbol table (to fix with SONARPY-647)
      .filter(symbol -> !isBuiltInOrImportedName(fullyQualifiedModuleName, symbol))
      .flatMap(s -> DescriptorUtils.descriptors(s).stream())
      .collect(Collectors.toList());

    modules.put(fullyQualifiedModuleName, new ModuleDescriptor(pythonFile.fileName(), fullyQualifiedModuleName, descriptors));
  }

  private static boolean isBuiltInOrImportedName(String fullyQualifiedModuleName, Symbol symbol) {
    String fullyQualifiedVariableName = symbol.fullyQualifiedName();
    return (fullyQualifiedVariableName != null && !fullyQualifiedVariableName.startsWith(fullyQualifiedModuleName)) ||
      symbol.usages().stream().anyMatch(u -> u.kind().equals(Usage.Kind.IMPORT));
  }

  public Map<String, ModuleDescriptor> modules() {
    return modules;
  }

}
