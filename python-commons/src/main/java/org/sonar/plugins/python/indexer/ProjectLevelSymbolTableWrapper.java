/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.indexer;

import org.sonar.api.scanner.ScannerSide;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonarsource.api.sonarlint.SonarLintSide;

/**
 * DI component that holds the project-level symbol table populated by {@code PythonSensor}.
 * Allows subsequent sensors (e.g. {@code PythonA3SContextCollectorSensor}) to access the
 * fully-populated symbol table without depending on {@link PythonIndexerWrapper}, which may
 * return {@code null} in the SonarQube Scanner path.
 *
 * <p>Defaults to {@link ProjectLevelSymbolTable#empty()} so consumers never receive {@code null}.
 */
@ScannerSide
@SonarLintSide
public class ProjectLevelSymbolTableWrapper {

  private ProjectLevelSymbolTable symbolTable = ProjectLevelSymbolTable.empty();

  public ProjectLevelSymbolTable symbolTable() {
    return symbolTable;
  }

  public void setSymbolTable(ProjectLevelSymbolTable symbolTable) {
    this.symbolTable = symbolTable;
  }
}
