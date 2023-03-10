/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import org.junit.Test;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.assertj.core.api.Assertions.assertThat;

public class TestPythonVisitorRunnerTest {

  @Test(expected = IllegalStateException.class)
  public void unknownFile() {
    TestPythonVisitorRunner.scanFile(new File("xxx"), visitorContext -> {});
  }

  @Test
  public void fileUri() throws IOException {
    File tmpFile = Files.createTempFile("foo", ".py").toFile();
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile);
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
  }

  @Test
  public void fileUriIPython() throws IOException {
    File tmpFile = Files.createTempFile("foo", ".ipynb").toFile();
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile);
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
  }

  @Test
  public void globalSymbols() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    List<File> files = List.of(new File(baseDir, "file.py"));
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(files, baseDir);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("file"))
      .extracting(Symbol::name)
      .containsExactlyInAnyOrder("hello", "A");
  }

  @Test
  public void globalSymbolsIPython() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    List<File> files = List.of(new File(baseDir, "file.py"), new File(baseDir, "file.ipynb"));
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(files, baseDir);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("file"))
      .extracting(Symbol::name)
      .containsExactlyInAnyOrder("hello", "A");
  }
}
