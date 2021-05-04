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
package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonIndexerTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/indexer").getAbsoluteFile();
  private SensorContextTester context;
  private static Path workDir;

  @Before
  public void init() throws IOException {
    context = SensorContextTester.create(baseDir);
    workDir = Files.createTempDirectory("workDir");
    context.fileSystem().setWorkDir(workDir);
  }

  @Test
  public void test_indexer() {
    PythonIndexer pythonIndexer = new PythonIndexer();
    InputFile file1 = inputFile("main.py");
    InputFile file2 = inputFile("mod.py");
    pythonIndexer.buildOnce(context, Arrays.asList(file1, file2));
    ProjectLevelSymbolTable projectLevelSymbolTable = pythonIndexer.projectLevelSymbolTable();
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).hasSize(1);
    Symbol modAddSymbol = projectLevelSymbolTable.getSymbol("mod.add");
    assertThat(modAddSymbol).isNotNull();
    assertThat(modAddSymbol.is(Symbol.Kind.FUNCTION)).isTrue();
  }

  @Test
  public void test_indexer_removed_file() {
    PythonIndexer pythonIndexer = new PythonIndexer();
    InputFile file1 = inputFile("main.py");
    InputFile file2 = inputFile("mod.py");
    pythonIndexer.buildOnce(context, Arrays.asList(file1, file2));
    ProjectLevelSymbolTable projectLevelSymbolTable = pythonIndexer.projectLevelSymbolTable();
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).hasSize(1);
    Symbol modAddSymbol = projectLevelSymbolTable.getSymbol("mod.add");
    assertThat(modAddSymbol).isNotNull();
    assertThat(modAddSymbol.is(Symbol.Kind.FUNCTION)).isTrue();

    pythonIndexer.removeFileFromProjectSymbolTable(file2);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).isNull();
    modAddSymbol = projectLevelSymbolTable.getSymbol("mod.add");
    assertThat(modAddSymbol).isNull();
  }

  @Test
  public void test_indexer_added_file() throws IOException {
    PythonIndexer pythonIndexer = new PythonIndexer();
    InputFile file1 = inputFile("main.py");
    InputFile file2 = inputFile("mod.py");
    pythonIndexer.buildOnce(context, Arrays.asList(file1, file2));
    ProjectLevelSymbolTable projectLevelSymbolTable = pythonIndexer.projectLevelSymbolTable();
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).hasSize(1);
    Symbol modAddSymbol = projectLevelSymbolTable.getSymbol("mod.add");
    assertThat(modAddSymbol).isNotNull();
    assertThat(modAddSymbol.is(Symbol.Kind.FUNCTION)).isTrue();

    InputFile file3 = inputFile("added.py");
    pythonIndexer.addFileToProjectSymbolTable(file3);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("added")).hasSize(1);
    Symbol newFuncSymbol = projectLevelSymbolTable.getSymbol("added.new_func");
    assertThat(newFuncSymbol).isNotNull();
    assertThat(newFuncSymbol.is(Symbol.Kind.FUNCTION)).isTrue();
  }

  private InputFile inputFile(String name) {
    DefaultInputFile inputFile = createInputFile(name);
    context.fileSystem().add(inputFile);
    return inputFile;
  }

  private DefaultInputFile createInputFile(String name) {
    return TestInputFileBuilder.create("moduleKey", name)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setType(InputFile.Type.MAIN)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, name), StandardCharsets.UTF_8))
      .build();
  }
}
