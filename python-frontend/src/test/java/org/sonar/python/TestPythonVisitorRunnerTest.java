/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TestPythonVisitorRunnerTest {

  @Test
  void unknownFile() {
    var file = new File("xxx");
    var check = (PythonCheck) visitorContext -> {};
    assertThatThrownBy(() -> TestPythonVisitorRunner.scanFile(file, check))
      .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void fileUri() throws IOException {
    File tmpFile = Files.createTempFile("foo", ".py").toFile();
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile);
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
  }

  @Test
  void fileUriIPython() throws IOException {
    File tmpFile = Files.createTempFile("foo", ".ipynb").toFile();
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile);
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
  }

  @Test
  void globalSymbols() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    List<File> files = List.of(new File(baseDir, "file.py"));
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(files, baseDir);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("file"))
      .extracting(Symbol::name)
      .containsExactlyInAnyOrder("hello", "A");
  }

  @Test
  void globalSymbolsIPython() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    List<File> files = List.of(new File(baseDir, "file.py"), new File(baseDir, "file.ipynb"));
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(files, baseDir);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("file"))
      .extracting(Symbol::name)
      .containsExactlyInAnyOrder("hello", "A");
  }

  @Test
  void scanNotebookFile() throws IOException {
    File tmpFile = Files.createTempFile("foo", ".ipynb").toFile();
    var check = (PythonCheck) visitorContect -> {};
    PythonVisitorContext context = TestPythonVisitorRunner.scanNotebookFile(tmpFile, Map.of(), "", check);
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
  }
}
