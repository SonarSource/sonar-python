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
package org.sonar.python.semantic;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.index.ModuleSummary;
import org.sonar.python.index.ProjectSummary;
import org.sonar.python.index.Summary;
import org.sonar.python.index.SummaryUtils;
import org.sonar.python.index.SymbolBuilder;

public class ProjectLevelSymbolTable {

  private final ProjectSummary projectSummary;

  public static ProjectLevelSymbolTable empty() {
    return new ProjectLevelSymbolTable(Collections.emptyMap());
  }

  public static ProjectLevelSymbolTable from(Map<String, Set<Symbol>> globalSymbolsByModuleName) {
    return new ProjectLevelSymbolTable(globalSymbolsByModuleName);
  }

  public ProjectLevelSymbolTable() {
    this.projectSummary = new ProjectSummary();
  }

  private ProjectLevelSymbolTable(Map<String, Set<Symbol>> globalSymbolsByModuleName) {
    Map<String, ModuleSummary> modules = new HashMap<>();
    for (Map.Entry<String, Set<Symbol>> entry : globalSymbolsByModuleName.entrySet()) {
      String moduleName = entry.getKey();
      Set<Symbol> exportedSymbols = entry.getValue();
      List<Summary> summaries = exportedSymbols.stream()
        .flatMap(s -> SummaryUtils.summary(s).stream())
        .collect(Collectors.toList());
      // TODO: Extract last dotted name from ModuleName?
      modules.put(moduleName, new ModuleSummary(moduleName, moduleName, summaries));
    }
    this.projectSummary = new ProjectSummary(modules);
  }

  public void addModule(FileInput fileInput, String packageName, PythonFile pythonFile) {
    projectSummary.addModule(fileInput, packageName, pythonFile);
  }

  @CheckForNull
  public Symbol getSymbol(@Nullable String fullyQualifiedName, Set<Symbol> existingSymbols) {
    return new SymbolBuilder(Collections.emptyMap(), projectSummary)
      .fromFullyQualifiedName(fullyQualifiedName)
      .build();
  }

  @CheckForNull
  public Set<Symbol> getSymbolsFromModule(@Nullable String moduleName,Set<Symbol> existingSymbols) {
    ModuleSummary moduleSummary = projectSummary.modules().get(moduleName);
    if (moduleSummary == null) {
      return null;
    }
    return moduleSummary.summariesByFQN().values().stream()
      .map(summaries -> new SymbolBuilder(Collections.emptyMap(), projectSummary).fromSummaries(summaries).build())
      .collect(Collectors.toSet());
  }

  public boolean isDjangoView(@Nullable String fqn) {
    return projectSummary.isDjangoView(fqn);
  }
}
