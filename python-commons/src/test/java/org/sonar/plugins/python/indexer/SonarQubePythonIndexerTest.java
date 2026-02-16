/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.plugins.python.PythonInputFile;
import org.sonar.plugins.python.api.caching.PythonReadCache;
import org.sonar.plugins.python.api.caching.PythonWriteCache;
import org.sonar.plugins.python.caching.TestReadCache;
import org.sonar.plugins.python.caching.TestWriteCache;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.caching.PythonReadCacheImpl;
import org.sonar.python.caching.PythonWriteCacheImpl;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.project.config.ProjectConfigurationBuilder;
import org.sonar.python.types.TypeShed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;
import static org.sonar.plugins.python.TestUtils.createInputFile;
import static org.sonar.plugins.python.caching.Caching.CACHE_VERSION_KEY;
import static org.sonar.plugins.python.caching.Caching.PROJECT_FILES_KEY;
import static org.sonar.plugins.python.caching.Caching.TYPESHED_MODULES_KEY;
import static org.sonar.plugins.python.caching.Caching.fileContentHashCacheKey;
import static org.sonar.plugins.python.caching.Caching.importsMapCacheKey;
import static org.sonar.plugins.python.caching.Caching.projectSymbolTableCacheKey;
import static org.sonar.python.index.DescriptorsToProtobuf.toProtobufModuleDescriptor;

class SonarQubePythonIndexerTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/indexer").getAbsoluteFile();
  private SensorContextTester context;

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  private PythonInputFile file1;
  private PythonInputFile file2;
  private SonarQubePythonIndexer pythonIndexer;
  private TestReadCache readCache;
  private TestWriteCache writeCache;
  private CacheContextImpl cacheContext;
  private String cacheVersion;

  @BeforeEach
  void init() throws IOException {
    TypeShed.resetBuiltinSymbols();
    context = SensorContextTester.create(baseDir);
    Path workDir = Files.createTempDirectory("workDir");
    context.fileSystem().setWorkDir(workDir);
    context.settings().setProperty("sonar.python.skipUnchanged", true);
    context.settings().setProperty("sonar.python.analysis.threads", 2);

    writeCache = new TestWriteCache();
    readCache = new TestReadCache();
    writeCache.bind(readCache);
    cacheVersion = "unknownPluginVersion";
    readCache.put(CACHE_VERSION_KEY, cacheVersion.getBytes(StandardCharsets.UTF_8));
    PythonWriteCache pythonWriteCache = new PythonWriteCacheImpl(writeCache);
    PythonReadCache pythonReadCache = new PythonReadCacheImpl(readCache);
    cacheContext = new CacheContextImpl(true, pythonWriteCache, pythonReadCache);
  }

  @Test
  void test_single_file_modified() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(importsMapCacheKey("moduleKey:main.py"), importsAsByteArray(List.of("mod")));
    readCache.put(importsMapCacheKey("moduleKey:mod.py"), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey("moduleKey:main.py"), serializedSymbolTable);
    readCache.put(projectSymbolTableCacheKey("moduleKey:mod.py"), outdatedEntry);
    readCache.put(fileContentHashCacheKey("moduleKey:main.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    readCache.put(fileContentHashCacheKey("moduleKey:mod.py"), file2.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isTrue();
    assertThat(logTester.logs(Level.INFO))
      .contains("Using cached data to retrieve global symbols.")
      .contains("Cached information of global symbols will be used for 1 out of 2 main files. Global symbols will be recomputed for the remaining files.")
      .contains("Fully optimized analysis can be performed for 1 out of 2 files.")
      .contains("1/1 source file has been analyzed");
    assertThat(logTester.logs(Level.WARN)).contains("""
      Implementation version of the Python plugin not found. Cached data may not be invalidated properly, \
      which may lead to inaccurate analysis results.""");
    assertThat(logTester.logs(Level.DEBUG))
      .contains("Cache version still up to date: \"unknownPluginVersion\".")
      .contains("Scanning global symbols in 2 threads");
  }

  @Test
  void test_modified_dependency() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(importsMapCacheKey("moduleKey:main.py"), importsAsByteArray(List.of("unknown", "mod", "other")));
    readCache.put(importsMapCacheKey("moduleKey:mod.py"), importsAsByteArray(Collections.emptyList()));
    readCache.put(projectSymbolTableCacheKey("moduleKey:main.py"), serializedSymbolTable);
    readCache.put(projectSymbolTableCacheKey("moduleKey:mod.py"), outdatedEntry);
    readCache.put(fileContentHashCacheKey("moduleKey:main.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    readCache.put(fileContentHashCacheKey("moduleKey:mod.py"), file2.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isTrue();
    assertThat(pythonIndexer.canBeFullyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO))
      .contains("Cached information of global symbols will be used for 1 out of 2 main files. Global symbols will be recomputed for the remaining files.")
      .contains("Fully optimized analysis can be performed for 0 out of 2 files.")
      .contains("Partially optimized analysis can be performed for 1 out of 2 files.")
      .contains("1/1 source file has been analyzed");
  }

  @Test
  void test_deleted_dependency() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(List.of(file1));

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(importsMapCacheKey("moduleKey:main.py"), importsAsByteArray(List.of("unknown", "mod", "other")));
    readCache.put(importsMapCacheKey("moduleKey:mod.py"), importsAsByteArray(Collections.emptyList()));
    readCache.put(PROJECT_FILES_KEY, importsAsByteArray(List.of("main", "mod")));
    readCache.put(projectSymbolTableCacheKey("moduleKey:main.py"), serializedSymbolTable);
    readCache.put(projectSymbolTableCacheKey("moduleKey:mod.py"), outdatedEntry);
    readCache.put(fileContentHashCacheKey("moduleKey:main.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isTrue();
    assertThat(pythonIndexer.canBeFullyScannedWithoutParsing(file1)).isFalse();
    assertThat(logTester.logs(Level.INFO))
      .contains("Cached information of global symbols will be used for 1 out of 1 main files. Global symbols will be recomputed for the remaining files.")
      .contains("Fully optimized analysis can be performed for 0 out of 1 files.")
      .contains("Partially optimized analysis can be performed for 1 out of 1 files.");

    byte[] bytes = writeCache.getData().get(PROJECT_FILES_KEY);
    HashSet<String> retrievedFileList = new HashSet<>(Arrays.asList(new String(bytes, StandardCharsets.UTF_8).split(";")));
    assertThat(retrievedFileList).containsExactlyInAnyOrder("main");
  }

  @Test
  void test_deleted_unrelated_file() {
    file1 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(List.of(file1));

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(importsMapCacheKey("moduleKey:main.py"), importsAsByteArray(List.of("unknown", "mod", "other")));
    readCache.put(importsMapCacheKey("moduleKey:mod.py"), importsAsByteArray(Collections.emptyList()));
    readCache.put(PROJECT_FILES_KEY, importsAsByteArray(List.of("main", "mod")));
    readCache.put(projectSymbolTableCacheKey("moduleKey:main.py"), serializedSymbolTable);
    readCache.put(projectSymbolTableCacheKey("moduleKey:mod.py"), outdatedEntry);
    readCache.put(fileContentHashCacheKey("moduleKey:mod.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isTrue();
    assertThat(logTester.logs(Level.INFO))
      .contains("Cached information of global symbols will be used for 1 out of 1 main files. Global symbols will be recomputed for the remaining files.")
      .contains("Fully optimized analysis can be performed for 1 out of 1 files.");

    byte[] bytes = writeCache.getData().get(PROJECT_FILES_KEY);
    HashSet<String> retrievedFileList = new HashSet<>(Arrays.asList(new String(bytes, StandardCharsets.UTF_8).split(";")));
    assertThat(retrievedFileList).containsExactlyInAnyOrder("mod");
  }

  @Test
  void test_no_file_modified_missing_entry() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    readCache.put(fileContentHashCacheKey("moduleKey:main.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    readCache.put(fileContentHashCacheKey("moduleKey:mod.py"), file2.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO))
      .contains("Cached information of global symbols will be used for 0 out of 2 main files. Global symbols will be recomputed for the remaining files.")
      .contains("Fully optimized analysis can be performed for 0 out of 2 files.")
      .contains("2/2 source files have been analyzed");
  }

  @Test
  void test_no_file_modified_missing_imports() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(projectSymbolTableCacheKey("moduleKey:main.py"), serializedSymbolTable);
    readCache.put(projectSymbolTableCacheKey("moduleKey:mod.py"), outdatedEntry);
    readCache.put(fileContentHashCacheKey("moduleKey:main.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    readCache.put(fileContentHashCacheKey("moduleKey:mod.py"), file2.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO))
      .contains("Cached information of global symbols will be used for 0 out of 2 main files. Global symbols will be recomputed for the remaining files.")
      .contains("Fully optimized analysis can be performed for 0 out of 2 files.")
      .contains("2/2 source files have been analyzed");
  }

  @Test
  void test_no_file_modified_missing_descriptors() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    readCache.put(importsMapCacheKey("moduleKey:main.py"), importsAsByteArray(List.of("mod")));
    readCache.put(importsMapCacheKey("moduleKey:mod.py"), importsAsByteArray(Collections.emptyList()));
    readCache.put(fileContentHashCacheKey("moduleKey:main.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    readCache.put(fileContentHashCacheKey("moduleKey:mod.py"), file2.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO))
      .contains("Cached information of global symbols will be used for 0 out of 2 main files. Global symbols will be recomputed for the remaining files.")
      .contains("Fully optimized analysis can be performed for 0 out of 2 files.")
      .contains("2/2 source files have been analyzed");
  }

  @Test
  void test_no_file_modified_invalid_cache_version() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    readCache.put(CACHE_VERSION_KEY, "outdatedVersion".getBytes(StandardCharsets.UTF_8));

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(importsMapCacheKey("moduleKey:main.py"), importsAsByteArray(List.of("mod")));
    readCache.put(importsMapCacheKey("moduleKey:mod.py"), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey("moduleKey:main.py"), serializedSymbolTable);
    readCache.put(projectSymbolTableCacheKey("moduleKey:mod.py"), outdatedEntry);
    readCache.put(fileContentHashCacheKey("moduleKey:main.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    readCache.put(fileContentHashCacheKey("moduleKey:mod.py"), file2.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));

    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO))
      .contains("""
        The cache version has changed since the previous analysis, cached data will not be used during this analysis. \
        Retrieved: "outdatedVersion". Current version: "unknownPluginVersion".""")
      .contains("2/2 source files have been analyzed");
  }

  @Test
  void test_no_file_modified_invalid_cache_version_due_to_changed_python_version() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    context.settings().setProperty("sonar.python.version", "3.11");
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    byte[] outdatedEntry = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(importsMapCacheKey("moduleKey:main.py"), importsAsByteArray(List.of("mod")));
    readCache.put(importsMapCacheKey("moduleKey:mod.py"), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey("moduleKey:main.py"), serializedSymbolTable);
    readCache.put(projectSymbolTableCacheKey("moduleKey:mod.py"), outdatedEntry);
    readCache.put(fileContentHashCacheKey("moduleKey:main.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    readCache.put(fileContentHashCacheKey("moduleKey:mod.py"), file2.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));

    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO))
      .contains("""
        The cache version has changed since the previous analysis, cached data will not be used during this analysis. \
        Retrieved: "unknownPluginVersion". Current version: "unknownPluginVersion;3.11".""")
      .contains("2/2 source files have been analyzed");
  }

  @Test
  void test_test_files_use_cache() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.TEST);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.CHANGED, InputFile.Type.TEST);
    readCache.put(fileContentHashCacheKey("moduleKey:main.py"), file1.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    readCache.put(importsMapCacheKey("moduleKey:main.py"), importsAsByteArray(List.of("mod")));
    readCache.put(projectSymbolTableCacheKey("moduleKey:main.py"), serializedSymbolTable);

    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isTrue();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO))
      .contains("Fully optimized analysis can be performed for 0 out of 2 files.")
      .contains("Partially optimized analysis can be performed for 1 out of 2 files.");
  }

  @Test
  void test_pr_analysis_disabled() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    context.settings().setProperty("sonar.python.skipUnchanged", false);
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO)).doesNotContain("Using cached data to retrieve global symbols.");
  }

  @Test
  void test_pr_analysis_enabled() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    SensorContext mockContext = spy(context);
    when(mockContext.canSkipUnchangedFiles()).thenReturn(true);
    context.settings().setProperty("sonar.python.skipUnchanged", false);
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(mockContext);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO)).contains("Using cached data to retrieve global symbols.");
  }

  @Test
  void test_disabled_cache() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(Arrays.asList(file1, file2));

    cacheContext = new CacheContextImpl(false, new PythonWriteCacheImpl(new TestWriteCache()), new PythonReadCacheImpl(new TestReadCache()));
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file2)).isFalse();
    assertThat(logTester.logs(Level.INFO)).doesNotContain("Using cached data to retrieve global symbols.");
  }

  @Test
  void test_typeshed_modules_not_cached_if_empty() {
    file1 = createInputFile(baseDir, "main.py", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(List.of(file1));

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
    assertThat(writeCache.getData()).doesNotContainKey(TYPESHED_MODULES_KEY);
  }

  @Test
  void test_regular_scan_when_scan_without_parsing_fails() {
    List<PythonInputFile> files = List.of(createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN));
    PythonIndexer.GlobalSymbolsScanner globalSymbolsScanner = spy(
      new SonarQubePythonIndexer(files, cacheContext, context, new ProjectConfigurationBuilder()).new GlobalSymbolsScanner(context));
    when(globalSymbolsScanner.canBeScannedWithoutParsing(any())).thenReturn(true);
    globalSymbolsScanner.execute(files, context);

    assertThat(logTester.logs(Level.INFO)).contains("1/1 source file has been analyzed");
  }

  @Test
  void test_no_data_in_cache_for_parse_error() {
    file1 = createInputFile(baseDir, "parse_error.py", InputFile.Status.ADDED, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(List.of(file1));

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);
    assertThat(writeCache.getData().containsKey(projectSymbolTableCacheKey("moduleKey:parse_error.py"))).isFalse();
  }

  @Test
  void test_file_content_hash_changed() {
    file1 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN);

    List<PythonInputFile> inputFiles = new ArrayList<>(List.of(file1));

    byte[] outdatedEntry = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("outdated", "mod.outdated", null))).toByteArray();
    readCache.put(importsMapCacheKey("moduleKey:mod.py"), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey("moduleKey:mod.py"), outdatedEntry);
    readCache.put(fileContentHashCacheKey("moduleKey:mod.py"), file2.wrappedFile().md5Hash().getBytes(StandardCharsets.UTF_8));
    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.canBePartiallyScannedWithoutParsing(file1)).isFalse();
  }

  @Test
  void test_notebook_should_not_be_in_project_level_symbol_table() {
    file1 = createInputFile(baseDir, "notebook.ipynb", InputFile.Status.SAME, InputFile.Type.MAIN);
    file2 = createInputFile(baseDir, "mod.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    List<PythonInputFile> inputFiles = new ArrayList<>(List.of(file1, file2));

    pythonIndexer = new SonarQubePythonIndexer(inputFiles, cacheContext, context, new ProjectConfigurationBuilder());
    pythonIndexer.buildOnce(context);

    assertThat(pythonIndexer.projectLevelSymbolTable().getSymbolsFromModule("mod")).isNotEmpty();
    assertThat(pythonIndexer.projectLevelSymbolTable().getSymbolsFromModule("notebook")).isEmpty();
  }

  @Test
  void test_sensor_single_thread() throws IOException {
    var contextSingleThread = SensorContextTester.create(baseDir);
    contextSingleThread.settings().setProperty("sonar.python.analysis.threads", 1);
    contextSingleThread.fileSystem().setWorkDir(Files.createTempDirectory("workDir"));

    var inputFiles = List.of(createInputFile(baseDir, "main.py", InputFile.Status.SAME, InputFile.Type.MAIN));
    var indexer = new SonarQubePythonIndexer(inputFiles, cacheContext, contextSingleThread, new ProjectConfigurationBuilder());
    indexer.buildOnce(contextSingleThread);

    assertThat(indexer.projectLevelSymbolTable().getSymbolsFromModule("main")).isNotEmpty();
  }

  private byte[] importsAsByteArray(List<String> mod) {
    return String.join(";", mod).getBytes(StandardCharsets.UTF_8);
  }

  // === Package Root Resolution Tests ===

  @Test
  void test_package_roots_from_pyproject_toml_setuptools() throws IOException {
    // Create a temp directory with pyproject.toml and source files
    Path tempDir = Files.createTempDirectory("pyproject_test").toRealPath();
    Path srcDir = tempDir.resolve(Path.of("src", "acme", "math", "stats"));
    Files.createDirectories(srcDir);
    Files.writeString(srcDir.resolve("__init__.py"), "");
    Files.writeString(srcDir.resolve("mean.py"), "def mean(): pass");
    Files.writeString(tempDir.resolve("pyproject.toml"), """
      [tool.setuptools.packages.find]
      where = ["src"]
      """);

    SensorContextTester tempContext = SensorContextTester.create(tempDir.toFile());
    tempContext.fileSystem().setWorkDir(Files.createTempDirectory("workDir"));
    tempContext.settings().setProperty("sonar.python.skipUnchanged", false);

    // Add pyproject.toml as input file so it can be found
    PythonInputFile pyprojectFile = createInputFile(tempDir.toFile(), "pyproject.toml", InputFile.Status.SAME, InputFile.Type.MAIN);
    tempContext.fileSystem().add(pyprojectFile.wrappedFile());

    PythonInputFile meanFile = createInputFile(srcDir.toFile(),
      "mean.py",
      InputFile.Status.ADDED, InputFile.Type.MAIN);
    List<PythonInputFile> inputFiles = List.of(meanFile);

    CacheContextImpl noCacheContext = new CacheContextImpl(false, new PythonWriteCacheImpl(new TestWriteCache()), new PythonReadCacheImpl(new TestReadCache()));
    SonarQubePythonIndexer indexer = new SonarQubePythonIndexer(inputFiles, noCacheContext, tempContext, new ProjectConfigurationBuilder());
    assertThat(indexer.packageRoots()).containsExactly(tempDir.resolve("src").toAbsolutePath().toString());

    // FQN should be computed correctly for namespace packages
    String packageName = indexer.packageName(meanFile);
    assertThat(packageName).isEqualTo("acme.math.stats");

    assertThat(logTester.logs(Level.DEBUG)).anyMatch(log -> log.contains("Resolved package roots from build configuration"));
  }

  @Test
  void test_package_roots_fallback_to_src_folder() throws IOException {
    // Create a temp directory with src folder but no pyproject.toml
    Path tempDir = Files.createTempDirectory("src_fallback_test");
    Path srcDir = tempDir.resolve(Path.of("src", "mypackage"));
    Files.createDirectories(srcDir);
    Files.writeString(srcDir.resolve("__init__.py"), "");
    Files.writeString(srcDir.resolve("module.py"), "x = 1");

    SensorContextTester tempContext = SensorContextTester.create(tempDir.toFile());
    tempContext.fileSystem().setWorkDir(Files.createTempDirectory("workDir"));
    tempContext.settings().setProperty("sonar.python.skipUnchanged", false);

    PythonInputFile moduleFile = createInputFile(tempDir.toFile(),
      "src/mypackage/module.py",
      InputFile.Status.ADDED, InputFile.Type.MAIN);
    List<PythonInputFile> inputFiles = List.of(moduleFile);

    CacheContextImpl noCacheContext = new CacheContextImpl(false, new PythonWriteCacheImpl(new TestWriteCache()), new PythonReadCacheImpl(new TestReadCache()));
    SonarQubePythonIndexer indexer = new SonarQubePythonIndexer(inputFiles, noCacheContext, tempContext, new ProjectConfigurationBuilder());

    // Package roots should fall back to src folder
    // Use File.getAbsolutePath() for both sides to ensure consistent path representation on Windows
    String expectedRoot = new File(tempDir.toFile(), "src").getAbsolutePath();
    assertThat(indexer.packageRoots()).containsExactly(expectedRoot);

    // FQN should be computed correctly
    String packageName = indexer.packageName(moduleFile);
    assertThat(packageName).isEqualTo("mypackage");

    assertThat(logTester.logs(Level.DEBUG)).anyMatch(log -> log.contains("Resolved package roots from fallback"));
  }

  @Test
  void test_package_roots_fallback_to_base_dir() throws IOException {
    // Create a temp directory without src folder or pyproject.toml
    Path tempDir = Files.createTempDirectory("basedir_fallback_test");
    Path pkgDir = Files.createDirectories(tempDir.resolve("mypackage"));
    Files.writeString(pkgDir.resolve("__init__.py"), "");
    Files.writeString(pkgDir.resolve("module.py"), "x = 1");

    SensorContextTester tempContext = SensorContextTester.create(tempDir.toFile());
    tempContext.fileSystem().setWorkDir(Files.createTempDirectory("workDir"));
    tempContext.settings().setProperty("sonar.python.skipUnchanged", false);

    PythonInputFile moduleFile = createInputFile(tempDir.toFile(),
      "mypackage/module.py",
      InputFile.Status.ADDED, InputFile.Type.MAIN);
    List<PythonInputFile> inputFiles = List.of(moduleFile);

    CacheContextImpl noCacheContext = new CacheContextImpl(false, new PythonWriteCacheImpl(new TestWriteCache()), new PythonReadCacheImpl(new TestReadCache()));
    SonarQubePythonIndexer indexer = new SonarQubePythonIndexer(inputFiles, noCacheContext, tempContext, new ProjectConfigurationBuilder());

    // Package roots should fall back to base dir (mypackage has __init__.py, so legacy detection works)
    // Use File.getAbsolutePath() for both sides to ensure consistent path representation on Windows
    String expectedRoot = tempDir.toFile().getAbsolutePath();
    assertThat(indexer.packageRoots()).containsExactly(expectedRoot);

    // FQN should be computed correctly using the base dir as root
    String packageName = indexer.packageName(moduleFile);
    assertThat(packageName).isEqualTo("mypackage");
  }

  @Test
  void test_package_roots_from_sonar_sources() throws IOException {
    // Create a temp directory with custom sources folder
    Path tempDir = Files.createTempDirectory("sonar_sources_test");
    Path libDir = tempDir.resolve(Path.of("lib", "mylib"));
    Files.createDirectories(libDir);
    Files.writeString(libDir.resolve("__init__.py"), "");
    Files.writeString(libDir.resolve("utils.py"), "x = 1");

    SensorContextTester tempContext = SensorContextTester.create(tempDir.toFile());
    tempContext.fileSystem().setWorkDir(Files.createTempDirectory("workDir"));
    tempContext.settings().setProperty("sonar.python.skipUnchanged", false);
    tempContext.settings().setProperty("sonar.sources", "lib");

    PythonInputFile utilsFile = createInputFile(tempDir.toFile(),
      "lib/mylib/utils.py",
      InputFile.Status.ADDED, InputFile.Type.MAIN);
    List<PythonInputFile> inputFiles = List.of(utilsFile);

    CacheContextImpl noCacheContext = new CacheContextImpl(false, new PythonWriteCacheImpl(new TestWriteCache()), new PythonReadCacheImpl(new TestReadCache()));
    SonarQubePythonIndexer indexer = new SonarQubePythonIndexer(inputFiles, noCacheContext, tempContext, new ProjectConfigurationBuilder());

    // Package roots should come from sonar.sources
    // Use File.getAbsolutePath() for both sides to ensure consistent path representation on Windows
    String expectedRoot = new File(tempDir.toFile(), "lib").getAbsolutePath();
    assertThat(indexer.packageRoots()).containsExactly(expectedRoot);

    // FQN should be computed correctly
    String packageName = indexer.packageName(utilsFile);
    assertThat(packageName).isEqualTo("mylib");
  }

  @Test
  void test_package_roots_from_setup_py() throws IOException {
    // Create a temp directory with setup.py and source files
    Path tempDir = Files.createTempDirectory("setup_py_test").toRealPath();
    Path srcDir = Files.createDirectories(tempDir.resolve(Path.of("src", "acme", "math", "stats")));
    Files.writeString(srcDir.resolve("__init__.py"), "");
    Files.writeString(srcDir.resolve("mean.py"), "def mean(): pass");
    Files.writeString(tempDir.resolve("setup.py"), """
      from setuptools import setup, find_packages
      setup(
          packages=find_packages(where="src"),
          package_dir={"": "src"}
      )
      """);

    SensorContextTester tempContext = SensorContextTester.create(tempDir.toFile());
    tempContext.fileSystem().setWorkDir(Files.createTempDirectory("workDir"));
    tempContext.settings().setProperty("sonar.python.skipUnchanged", false);

    // Add setup.py as input file so it can be found
    PythonInputFile setupPyFile = createInputFile(tempDir.toFile(), "setup.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    tempContext.fileSystem().add(setupPyFile.wrappedFile());

    PythonInputFile meanFile = createInputFile(tempDir.toFile(),
      Path.of("src", "acme", "math", "stats", "mean.py").toString(),
      InputFile.Status.ADDED, InputFile.Type.MAIN);
    List<PythonInputFile> inputFiles = List.of(meanFile);

    CacheContextImpl noCacheContext = new CacheContextImpl(false, new PythonWriteCacheImpl(new TestWriteCache()), new PythonReadCacheImpl(new TestReadCache()));
    SonarQubePythonIndexer indexer = new SonarQubePythonIndexer(inputFiles, noCacheContext, tempContext, new ProjectConfigurationBuilder());

    // Package roots should be resolved from setup.py
    assertThat(indexer.packageRoots()).containsExactly(tempDir.resolve("src").toAbsolutePath().toString());

    // FQN should be computed correctly for namespace packages
    String packageName = indexer.packageName(meanFile);
    assertThat(packageName).isEqualTo("acme.math.stats");

    assertThat(logTester.logs(Level.DEBUG)).anyMatch(log -> log.contains("Resolved package roots from build configuration"));
  }

  @Test
  void test_package_roots_from_both_pyproject_and_setup_py() throws IOException {
    // Create a temp directory with both pyproject.toml and setup.py
    Path tempDir = Files.createTempDirectory("both_configs_test").toRealPath();
    Path srcDir = Files.createDirectories(tempDir.resolve(Path.of("src", "mypackage")));
    Path libDir = Files.createDirectories(tempDir.resolve(Path.of("lib", "otherpackage")));
    Files.writeString(srcDir.resolve("__init__.py"), "");
    Files.writeString(libDir.resolve("__init__.py"), "");
    Files.writeString(srcDir.resolve("module.py"), "x = 1");
    Files.writeString(libDir.resolve("utils.py"), "y = 2");

    // pyproject.toml specifies src
    Files.writeString(tempDir.resolve("pyproject.toml"), """
      [tool.setuptools.packages.find]
      where = ["src"]
      """);

    // setup.py specifies lib
    Files.writeString(tempDir.resolve("setup.py"), """
      from setuptools import setup
      setup(package_dir={"": "lib"})
      """);

    SensorContextTester tempContext = SensorContextTester.create(tempDir.toFile());
    tempContext.fileSystem().setWorkDir(Files.createTempDirectory("workDir"));
    tempContext.settings().setProperty("sonar.python.skipUnchanged", false);

    // Add both config files
    PythonInputFile pyprojectFile = createInputFile(tempDir.toFile(), "pyproject.toml", InputFile.Status.SAME, InputFile.Type.MAIN);
    PythonInputFile setupPyFile = createInputFile(tempDir.toFile(), "setup.py", InputFile.Status.SAME, InputFile.Type.MAIN);
    tempContext.fileSystem().add(pyprojectFile.wrappedFile());
    tempContext.fileSystem().add(setupPyFile.wrappedFile());

    PythonInputFile moduleFile = createInputFile(tempDir.toFile(),
      Path.of("src", "mypackage", "module.py").toString(),
      InputFile.Status.ADDED, InputFile.Type.MAIN);
    PythonInputFile utilsFile = createInputFile(tempDir.toFile(),
      Path.of("lib", "otherpackage", "utils.py").toString(),
      InputFile.Status.ADDED, InputFile.Type.MAIN);
    List<PythonInputFile> inputFiles = List.of(moduleFile, utilsFile);

    CacheContextImpl noCacheContext = new CacheContextImpl(false, new PythonWriteCacheImpl(new TestWriteCache()), new PythonReadCacheImpl(new TestReadCache()));
    SonarQubePythonIndexer indexer = new SonarQubePythonIndexer(inputFiles, noCacheContext, tempContext, new ProjectConfigurationBuilder());

    // Package roots should include both src and lib
    assertThat(indexer.packageRoots()).containsExactlyInAnyOrder(
      tempDir.resolve("src").toAbsolutePath().toString(),
      tempDir.resolve("lib").toAbsolutePath().toString());

    // FQN should be computed correctly for both
    assertThat(indexer.packageName(moduleFile)).isEqualTo("mypackage");
    assertThat(indexer.packageName(utilsFile)).isEqualTo("otherpackage");
  }
}
