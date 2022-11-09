/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.utils.log.LogTester;
import org.sonar.api.utils.log.LoggerLevel;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.caching.Caching;
import org.sonar.python.caching.PythonReadCacheImpl;
import org.sonar.python.caching.PythonWriteCacheImpl;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;
import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.TestUtils.createInputFile;
import static org.sonar.python.caching.Caching.IMPORTS_MAP_CACHE_KEY_PREFIX;
import static org.sonar.python.caching.Caching.PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX;

public class SonarQubePythonIndexerTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/indexer").getAbsoluteFile();
  private SensorContextTester context;

  @org.junit.Rule
  public LogTester logTester = new LogTester();

  InputFile file1;
  InputFile file2;
  TestModuleFileSystem moduleFileSystem;
  SonarQubePythonIndexer pythonIndexer;
  ProjectLevelSymbolTable projectLevelSymbolTable;
  TestReadCache readCache;
  TestWriteCache writeCache;
  CacheContextImpl cacheContext;

  @Before
  public void init() throws IOException {
    context = SensorContextTester.create(baseDir);
    Path workDir = Files.createTempDirectory("workDir");
    context.fileSystem().setWorkDir(workDir);
    context.settings().setProperty("sonar.python.skipUnchanged", true);

    writeCache = new TestWriteCache();
    readCache = new TestReadCache();
    writeCache.bind(readCache);
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);
  }

  @Test
  public void test_single_file_modified() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.CHANGED);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME);

    List<InputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));
    moduleFileSystem = new TestModuleFileSystem(inputFiles);

    byte[] serializedSymbolTable = Caching.moduleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = Caching.moduleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(IMPORTS_MAP_CACHE_KEY_PREFIX + "main", String.join(";", List.of("mod")).getBytes(StandardCharsets.UTF_8));
    readCache.put(IMPORTS_MAP_CACHE_KEY_PREFIX + "mod", String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "main", serializedSymbolTable);
    readCache.put(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod", outdatedEntry);
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext);
    pythonIndexer.buildOnce(context);
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file2)).isTrue();
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Retrieving cached project level symbol table.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Project level symbol table information needs to be computed for 1 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Regular analysis will be performed on 1 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("1/1 source file has been analyzed");
  }

  @Test
  public void test_modified_dependency() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.CHANGED);

    List<InputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));
    moduleFileSystem = new TestModuleFileSystem(inputFiles);

    byte[] serializedSymbolTable = Caching.moduleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = Caching.moduleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(IMPORTS_MAP_CACHE_KEY_PREFIX + "main", String.join(";", List.of("mod")).getBytes(StandardCharsets.UTF_8));
    readCache.put(IMPORTS_MAP_CACHE_KEY_PREFIX + "mod", String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "main", serializedSymbolTable);
    readCache.put(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod", outdatedEntry);
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext);
    pythonIndexer.buildOnce(context);
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Project level symbol table information needs to be computed for 1 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Regular analysis will be performed on 2 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("1/1 source file has been analyzed");
  }


  @Test
  public void test_no_file_modified_missing_entry() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME);

    List<InputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));
    moduleFileSystem = new TestModuleFileSystem(inputFiles);

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext);
    pythonIndexer.buildOnce(context);
    projectLevelSymbolTable = pythonIndexer.projectLevelSymbolTable();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Project level symbol table information needs to be computed for 2 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Regular analysis will be performed on 2 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("2/2 source files have been analyzed");
  }

  @Test
  public void test_no_file_modified_missing_imports() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME);

    List<InputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));
    moduleFileSystem = new TestModuleFileSystem(inputFiles);

    byte[] serializedSymbolTable = Caching.moduleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = Caching.moduleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "main", serializedSymbolTable);
    readCache.put(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "mod", outdatedEntry);

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext);
    pythonIndexer.buildOnce(context);
    projectLevelSymbolTable = pythonIndexer.projectLevelSymbolTable();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Project level symbol table information needs to be computed for 2 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Regular analysis will be performed on 2 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("2/2 source files have been analyzed");
  }

  @Test
  public void test_no_file_modified_missing_descriptors() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME);

    List<InputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));
    moduleFileSystem = new TestModuleFileSystem(inputFiles);

    readCache.put(IMPORTS_MAP_CACHE_KEY_PREFIX + "main", String.join(";", List.of("mod")).getBytes(StandardCharsets.UTF_8));
    readCache.put(IMPORTS_MAP_CACHE_KEY_PREFIX + "mod", String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext);
    pythonIndexer.buildOnce(context);
    projectLevelSymbolTable = pythonIndexer.projectLevelSymbolTable();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Project level symbol table information needs to be computed for 2 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Regular analysis will be performed on 2 out of 2 files.");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("2/2 source files have been analyzed");
  }

  @Test
  public void test_pr_analysis_disabled() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.CHANGED);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME);

    List<InputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));
    moduleFileSystem = new TestModuleFileSystem(inputFiles);

    context.settings().setProperty("sonar.python.skipUnchanged", false);
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext);
    pythonIndexer.buildOnce(context);
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(LoggerLevel.INFO)).doesNotContain("Retrieving cached project level symbol table.");
  }

  @Test
  public void test_disabled_cache() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.CHANGED);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME);

    List<InputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));
    moduleFileSystem = new TestModuleFileSystem(inputFiles);

    cacheContext = new CacheContextImpl(false, new PythonWriteCacheImpl(new TestWriteCache()), new PythonReadCacheImpl(new TestReadCache()));
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext);
    pythonIndexer.buildOnce(context);
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBeScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(LoggerLevel.INFO)).doesNotContain("Retrieving cached project level symbol table.");
  }

  @Test
  public void test_regular_scan_when_scan_without_parsing_fails()  {
    List<InputFile> files = List.of(createInputFile(baseDir, "main.py", InputFile.Status.SAME));
    PythonIndexer.GlobalSymbolsScanner globalSymbolsScanner = spy(
      new SonarQubePythonIndexer(files, cacheContext). new GlobalSymbolsScanner(context)
    );
    when(globalSymbolsScanner.canBeScannedWithoutParsing(any())).thenReturn(true);
    globalSymbolsScanner.execute(files, context);
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("1/1 source file has been analyzed");
  }
}
