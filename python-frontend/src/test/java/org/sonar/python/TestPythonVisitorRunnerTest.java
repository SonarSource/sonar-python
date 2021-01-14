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
package org.sonar.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collections;
import org.junit.Test;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.assertj.core.api.Assertions.assertThat;

public class TestPythonVisitorRunnerTest {

  @Test(expected = IllegalStateException.class)
  public void unknown_file() {
    TestPythonVisitorRunner.scanFile(new File("xxx"), visitorContext -> {});
  }

  @Test
  public void file_uri() throws IOException {
    File tmpFile = Files.createTempFile("foo", "py").toFile();
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile);
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
  }

  @Test
  public void global_symbols() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(Collections.singletonList(new File(baseDir, "file.py")), baseDir);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("file")).extracting(Symbol::name).containsExactlyInAnyOrder("hello", "A");
  }
}
