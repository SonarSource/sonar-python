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
package org.sonar.plugins.python.indexer;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.PythonInputFile;
import org.sonar.plugins.python.PythonInputFileImpl;
import org.sonar.plugins.python.TestUtils;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.assertj.core.api.Assertions.assertThat;

class SonarLintPythonIndexerProjectLevelSymbolTableBuildingTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/indexer/v2").getAbsoluteFile();

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @Test
  void single_file_simple_test() throws IOException {
    var projectLevelSymbolTable = buildProjectLevelSymbolTable("script.py");
    assertThat(projectLevelSymbolTable.getDescriptorsFromModule("script")).hasSize(4);
    Set<Descriptor> moduleDescriptors = projectLevelSymbolTable.getDescriptorsFromModuleV2("script");
    assertThat(moduleDescriptors).hasSize(4);

    var aClassDescriptor = moduleDescriptors
      .stream()
      .filter(d -> d.name().equals("A"))
      .findFirst()
      .filter(ClassDescriptor.class::isInstance)
      .map(ClassDescriptor.class::cast)
      .orElse(null);
    assertThat(aClassDescriptor).isNotNull();
    assertThat(aClassDescriptor.members()).hasSize(1);
    assertThat(aClassDescriptor.superClasses()).containsOnly("script.Parent", "int");

    var doSomethingDescriptor = aClassDescriptor.members()
      .stream()
      .filter(d -> d.name().equals("do_something"))
      .findFirst()
      .filter(FunctionDescriptor.class::isInstance)
      .map(FunctionDescriptor.class::cast)
      .orElse(null);
    assertThat(doSomethingDescriptor).isNotNull();
    assertThat(doSomethingDescriptor.parameters()).hasSize(2);
  }

  @Test
  void multiple_files_simple_test() throws IOException {
    var projectLevelSymbolTable = buildProjectLevelSymbolTable("mod1.py", "mod2.py");
    assertThat(projectLevelSymbolTable.getDescriptorsFromModule("mod1")).hasSize(1);
    assertThat(projectLevelSymbolTable.getDescriptorsFromModuleV2("mod2")).hasSize(1);
  }

  private ProjectLevelSymbolTable buildProjectLevelSymbolTable(String... files) throws IOException {
    var context = SensorContextTester.create(baseDir);
    var workDir = Files.createTempDirectory("workDir");
    context.fileSystem().setWorkDir(workDir);
    var inputFiles = Stream.of(files)
      .map(fileName -> inputFile(context, fileName))
      .toList();
    var moduleFileSystem = new TestModuleFileSystem(inputFiles);
    var pythonIndexer = new SonarLintPythonIndexer(moduleFileSystem);
    pythonIndexer.buildOnce(context);
    return pythonIndexer.projectLevelSymbolTable();
  }

  private PythonInputFile inputFile(SensorContextTester context, String name) {
    var inputFile = createInputFile(name);
    context.fileSystem().add(inputFile.wrappedFile());
    return inputFile;
  }

  private PythonInputFile createInputFile(String name) {
    return createInputFile(name, Python.KEY);
  }

  private PythonInputFile createInputFile(String name, String languageKey) {
    return new PythonInputFileImpl(TestInputFileBuilder.create("moduleKey", name)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setType(InputFile.Type.MAIN)
      .setLanguage(languageKey)
      .initMetadata(TestUtils.fileContent(new File(baseDir, name), StandardCharsets.UTF_8))
      .build());
  }
}
