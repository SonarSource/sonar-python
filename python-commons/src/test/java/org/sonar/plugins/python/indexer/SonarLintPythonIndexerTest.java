/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.plugins.python.indexer;

import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.jupiter.api.BeforeEach;
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
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.caching.DummyCache;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileEvent;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class SonarLintPythonIndexerTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/indexer").getAbsoluteFile();
  private SensorContextTester context;

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  PythonInputFile file1;
  PythonInputFile file2;
  TestModuleFileSystem moduleFileSystem;
  SonarLintPythonIndexer pythonIndexer;
  ProjectLevelSymbolTable projectLevelSymbolTable;

  @BeforeEach
  void init() throws IOException {
    context = SensorContextTester.create(baseDir);
    Path workDir = Files.createTempDirectory("workDir");
    context.fileSystem().setWorkDir(workDir);

    file1 = inputFile("main.py");
    file2 = inputFile("mod.py");
    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));
    moduleFileSystem = new TestModuleFileSystem(inputFiles);
    pythonIndexer = new SonarLintPythonIndexer(moduleFileSystem);
    pythonIndexer.buildOnce(context);
    projectLevelSymbolTable = pythonIndexer.projectLevelSymbolTable();
  }

  @Test
  void test_indexer() {
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).hasSize(1);
    Symbol modAddSymbol = projectLevelSymbolTable.getSymbol("mod.add");
    assertThat(modAddSymbol).isNotNull();
    assertThat(modAddSymbol.is(Symbol.Kind.FUNCTION)).isTrue();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBeFullyScannedWithoutParsing(file1)).isFalse();
  }

  @Test
  void build_once_should_build_once() {
    PythonInputFile file3 = inputFile("added.py");
    moduleFileSystem.addFile(file3);
    pythonIndexer.buildOnce(context);

    assertThat(projectLevelSymbolTable.getSymbolsFromModule("added")).isNull();
    assertThat(projectLevelSymbolTable.getSymbol("added.new_func")).isNull();
  }

  @Test
  void test_indexer_removed_file() {
    ModuleFileEvent moduleFileEvent = mock(ModuleFileEvent.class);
    when(moduleFileEvent.getType()).thenReturn(ModuleFileEvent.Type.DELETED);
    when(moduleFileEvent.getTarget()).thenReturn(file2.wrappedFile());
    pythonIndexer.process(moduleFileEvent);

    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).isNull();
    Symbol modAddSymbol = projectLevelSymbolTable.getSymbol("mod.add");
    assertThat(modAddSymbol).isNull();
  }

  @Test
  void test_indexer_file_removed_twice() {
    ModuleFileEvent moduleFileEvent = mock(ModuleFileEvent.class);
    when(moduleFileEvent.getType()).thenReturn(ModuleFileEvent.Type.DELETED);
    when(moduleFileEvent.getTarget()).thenReturn(file2.wrappedFile());
    pythonIndexer.process(moduleFileEvent);

    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).isNull();
    pythonIndexer.process(moduleFileEvent);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).isNull();

  }

  @Test
  void test_indexer_added_file() throws IOException {
    PythonInputFile file3 = createInputFile("added.py");
    ModuleFileEvent moduleFileEvent = mock(ModuleFileEvent.class);
    when(moduleFileEvent.getType()).thenReturn(ModuleFileEvent.Type.CREATED);
    when(moduleFileEvent.getTarget()).thenReturn(file3.wrappedFile());
    pythonIndexer.process(moduleFileEvent);

    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("added")).hasSize(1);
    Symbol newFuncSymbol = projectLevelSymbolTable.getSymbol("added.new_func");
    assertThat(newFuncSymbol).isNotNull();
    assertThat(newFuncSymbol.is(Symbol.Kind.FUNCTION)).isTrue();
  }

  @Test
  void test_indexer_added_nonexistent_file() throws IOException {
    InputFile nonExistentFile = TestInputFileBuilder.create("moduleKey", "nonexistent.py")
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setType(InputFile.Type.MAIN)
      .setLanguage(Python.KEY)
      .build();
    ModuleFileEvent moduleFileEvent = mock(ModuleFileEvent.class);
    when(moduleFileEvent.getType()).thenReturn(ModuleFileEvent.Type.CREATED);
    when(moduleFileEvent.getTarget()).thenReturn(nonExistentFile);
    pythonIndexer.process(moduleFileEvent);

    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("nonexistent")).isNull();
    assertThat(logTester.logs(Level.DEBUG)).contains("Failed to load file \"nonexistent.py\" (CREATED) to the project symbol table");
  }

  @Test
  void test_indexer_modified_file() throws IOException {
    ModuleFileEvent moduleFileEvent = mock(ModuleFileEvent.class);
    when(moduleFileEvent.getType()).thenReturn(ModuleFileEvent.Type.MODIFIED);
    when(moduleFileEvent.getTarget()).thenReturn(file2.wrappedFile());
    pythonIndexer.process(moduleFileEvent);

    assertThat(projectLevelSymbolTable.getSymbolsFromModule("main")).hasSize(1);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("mod")).hasSize(1);
    Symbol modAddSymbol = projectLevelSymbolTable.getSymbol("mod.add");
    assertThat(modAddSymbol).isNotNull();
  }

  @Test
  void test_indexer_non_python_file() {
    testNonPythonFile("txt");
    testNonPythonFile(null);
  }

  @Test
  void test_sonarlint_cache() throws IOException {
    PythonIndexer indexer = new SonarLintPythonIndexer(moduleFileSystem);
    CacheContext cacheContext = indexer.cacheContext();
    assertThat(cacheContext.isCacheEnabled()).isFalse();
    assertThat(cacheContext.getWriteCache()).isInstanceOf(DummyCache.class);
    assertThat(cacheContext.getReadCache()).isInstanceOf(DummyCache.class);

    indexer.setSonarLintCache(null);
    cacheContext = indexer.cacheContext();
    assertThat(cacheContext.isCacheEnabled()).isFalse();
    assertThat(cacheContext.getWriteCache()).isInstanceOf(DummyCache.class);
    assertThat(cacheContext.getReadCache()).isInstanceOf(DummyCache.class);

    SonarLintCache sonarLintCache = new SonarLintCache();
    indexer.setSonarLintCache(sonarLintCache);
    cacheContext = indexer.cacheContext();
    assertThat(cacheContext.isCacheEnabled()).isTrue();
    assertThat(cacheContext.getWriteCache()).isInstanceOf(PythonWriteCache.class);
    assertThat(cacheContext.getReadCache()).isInstanceOf(PythonReadCache.class);

    byte[] bytes = {0};
    sonarLintCache.write("foo", bytes);
    PythonReadCache readCache = cacheContext.getReadCache();
    try (var inputStream = readCache.read("foo")) {
      assertThat(inputStream.readAllBytes()).isEqualTo(bytes);
    }
  }

  private void testNonPythonFile(@Nullable String language) {
    ModuleFileEvent moduleFileEvent = mock(ModuleFileEvent.class);
    PythonInputFile txtFile = createInputFile("non_python.txt", language);
    when(moduleFileEvent.getTarget()).thenReturn(txtFile.wrappedFile());
    try {
      pythonIndexer.process(moduleFileEvent);
    } catch (RecognitionException exception) {
      fail("Non Python files should not be parsed.");
    }
    assertThat(logTester.logs(Level.DEBUG)).contains("Module file event for non_python.txt has been ignored because it's not a Python file.");
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("non_python")).isNull();
  }

  private PythonInputFile inputFile(String name) {
    PythonInputFile inputFile = createInputFile(name);
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
