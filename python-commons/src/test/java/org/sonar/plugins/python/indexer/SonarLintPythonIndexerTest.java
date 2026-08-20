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
import org.junit.jupiter.api.io.TempDir;
import org.slf4j.event.Level;
import org.sonar.api.batch.fs.InputFile;
import com.sonarsource.scanner.engine.sensor.test.fixtures.TestInputFileBuilder;
import com.sonarsource.scanner.engine.sensor.test.fixtures.SensorContextTester;
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
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.caching.DummyCache;
import org.sonar.python.project.config.ProjectConfigurationBuilder;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.scanner.plugin.api.impl.config.MapSettings;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileEvent;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
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
    pythonIndexer = new SonarLintPythonIndexer(moduleFileSystem, new ProjectConfigurationBuilder());
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
  void test_indexer_added_file() {
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
  void test_indexer_added_nonexistent_file() {
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
  void test_indexer_modified_file() {
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
  void incremental_update_preserves_build_config_roots(@TempDir Path projectDir) throws IOException {
    Files.createDirectories(projectDir.resolve("src/acme"));
    Files.writeString(projectDir.resolve("setup.py"), "setup(package_dir={'': 'src'}, packages=['acme'])");
    PythonInputFile module = inputFile(projectDir, "src/acme/mod.py", "def original():\n    pass\n");
    var localContext = SensorContextTester.create(projectDir.toFile());
    localContext.fileSystem().add(module.wrappedFile());
    var localIndexer = new SonarLintPythonIndexer(
      new TestModuleFileSystem(new ArrayList<>(List.of(module))), new ProjectConfigurationBuilder());

    localIndexer.buildOnce(localContext);

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.resolve("src").toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("acme.mod.original")).isNotNull();

    PythonInputFile added = inputFile(projectDir, "src/acme/added.py", "def added():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, added));

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.resolve("src").toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("acme.added.added")).isNotNull();
  }

  @Test
  void modified_package_configuration_file_refreshes_roots(@TempDir Path projectDir) throws IOException {
    Path pyproject = projectDir.resolve("pyproject.toml");
    Files.writeString(pyproject, "[project]\nname = 'before'\n");
    PythonInputFile module = inputFile(projectDir, "module.py", "def original():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, module));

    String updatedConfig = "[project]\nname = 'after'\n";
    Files.writeString(pyproject, updatedConfig);
    InputFile configurationFile = TestInputFileBuilder.create("moduleKey", "pyproject.toml")
      .setModuleBaseDir(projectDir)
      .setCharset(StandardCharsets.UTF_8)
      .setType(InputFile.Type.MAIN)
      .setLanguage(null)
      .initMetadata(updatedConfig)
      .build();

    localIndexer.process(event(ModuleFileEvent.Type.MODIFIED, configurationFile));

    verify(localIndexer).refreshPackageRoots();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.original")).isNotNull();
  }

  @Test
  void incremental_init_file_addition_rebuilds_modules_with_new_fqns(@TempDir Path projectDir) throws IOException {
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def foo():\n    pass\n");
    var localContext = SensorContextTester.create(projectDir.toFile());
    localContext.fileSystem().add(module.wrappedFile());
    var localIndexer = spy(new SonarLintPythonIndexer(
      new TestModuleFileSystem(new ArrayList<>(List.of(module))), new ProjectConfigurationBuilder()));
    localIndexer.buildOnce(localContext);

    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.foo")).isNotNull();

    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, packageInit));

    verify(localIndexer).refreshPackageRoots();
    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.foo")).isNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.module.foo")).isNotNull();
  }

  @Test
  void incremental_init_file_deletion_rebuilds_modules_with_new_fqns(@TempDir Path projectDir) throws IOException {
    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def foo():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, packageInit, module));

    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.module.foo")).isNotNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.foo")).isNull();

    Files.delete(packageInit.wrappedFile().path());

    localIndexer.process(event(ModuleFileEvent.Type.DELETED, packageInit));

    verify(localIndexer).refreshPackageRoots();
    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.resolve("pkg").toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.module.foo")).isNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.foo")).isNotNull();
  }

  @Test
  void created_file_under_existing_root_adds_more_specific_legacy_root(@TempDir Path projectDir) throws IOException {
    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def original():\n    pass\n");
    var localIndexer = buildIndexer(projectDir, packageInit, module);

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.toString());

    PythonInputFile script = inputFile(projectDir, "scripts/tool.py", "def run():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, script));

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.resolve("scripts").toString(), projectDir.toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("tool.run")).isNotNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("scripts.tool.run")).isNull();
  }

  @Test
  void created_file_with_existing_candidate_root_skips_root_resolution(@TempDir Path projectDir) throws IOException {
    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def original():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, packageInit, module));

    PythonInputFile added = inputFile(projectDir, "pkg/added.py", "def added():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, added));

    verify(localIndexer, never()).refreshPackageRoots();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.added.added")).isNotNull();
  }

  @Test
  void created_file_in_new_conventional_folder_refreshes_when_candidate_root_is_active(@TempDir Path projectDir) throws IOException {
    Files.writeString(projectDir.resolve("pyproject.toml"), "[project]\nname = 'sample'\n");
    PythonInputFile main = inputFile(projectDir, "main.py", "def original():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, main));
    Files.createDirectories(projectDir.resolve("src"));
    Files.writeString(projectDir.resolve("src/__init__.py"), "");

    PythonInputFile added = inputFile(projectDir, "src/tool.py", "def run():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, added));

    verify(localIndexer).refreshPackageRoots();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("src.tool.run")).isNotNull();
  }

  @Test
  void created_file_covered_by_base_dir_fallback_skips_root_resolution(@TempDir Path projectDir) throws IOException {
    Files.writeString(projectDir.resolve("pyproject.toml"), "[project]\nname = 'sample'\n");
    PythonInputFile main = inputFile(projectDir, "main.py", "def original():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, main));

    PythonInputFile added = inputFile(projectDir, "scripts/tool.py", "def run():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, added));

    verify(localIndexer, never()).refreshPackageRoots();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("scripts.tool.run")).isNotNull();
  }

  @Test
  void deleted_file_covered_by_base_dir_fallback_skips_root_resolution(@TempDir Path projectDir) throws IOException {
    Files.writeString(projectDir.resolve("pyproject.toml"), "[project]\nname = 'sample'\n");
    PythonInputFile first = inputFile(projectDir, "scripts/first.py", "def first():\n    pass\n");
    PythonInputFile second = inputFile(projectDir, "scripts/second.py", "def second():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, first, second));
    Files.delete(first.wrappedFile().path());

    localIndexer.process(event(ModuleFileEvent.Type.DELETED, first));

    verify(localIndexer, never()).refreshPackageRoots();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("scripts.first.first")).isNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("scripts.second.second")).isNotNull();
  }

  @Test
  void deleted_file_covered_by_configured_root_skips_root_resolution(@TempDir Path projectDir) throws IOException {
    Files.writeString(projectDir.resolve("setup.py"), "setup(package_dir={'': 'src'}, packages=['acme'])");
    PythonInputFile module = inputFile(projectDir, "src/acme/module.py", "def original():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, module));

    PythonInputFile added = inputFile(projectDir, "src/acme/added.py", "def added():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, added));
    clearInvocations(localIndexer);
    Files.delete(added.wrappedFile().path());

    localIndexer.process(event(ModuleFileEvent.Type.DELETED, added));

    verify(localIndexer, never()).refreshPackageRoots();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("acme.added.added")).isNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("acme.module.original")).isNotNull();
  }

  @Test
  void test_file_root_survives_refresh_triggered_by_main_file(@TempDir Path projectDir) throws IOException {
    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def original():\n    pass\n");
    PythonInputFile testFile = inputFile(projectDir, "tests/unit/test_only.py", "def test_something():\n    pass\n", InputFile.Type.TEST);
    var localIndexer = buildIndexer(projectDir, packageInit, module, testFile);

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.resolve("tests/unit").toString(), projectDir.toString());
    assertThat(localIndexer.getFileWithId(testFile.wrappedFile().absolutePath())).isNull();

    PythonInputFile script = inputFile(projectDir, "scripts/tool.py", "def run():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, script));

    assertThat(localIndexer.packageRoots()).containsExactly(
      projectDir.resolve("tests/unit").toString(),
      projectDir.resolve("scripts").toString(),
      projectDir.toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.module.original")).isNotNull();
    assertThat(localIndexer.getFileWithId(testFile.wrappedFile().absolutePath())).isNull();
  }

  @Test
  void created_test_file_updates_roots_without_entering_project_index(@TempDir Path projectDir) throws IOException {
    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def original():\n    pass\n");
    var localIndexer = buildIndexer(projectDir, packageInit, module);

    PythonInputFile testFile = inputFile(projectDir, "tests/unit/test_only.py", "def test_something():\n    pass\n", InputFile.Type.TEST);
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, testFile));

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.resolve("tests/unit").toString(), projectDir.toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.module.original")).isNotNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("test_only.test_something")).isNull();
    assertThat(localIndexer.getFileWithId(testFile.wrappedFile().absolutePath())).isNull();
  }

  @Test
  void deleted_test_file_removes_its_root_without_affecting_main_index(@TempDir Path projectDir) throws IOException {
    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def original():\n    pass\n");
    PythonInputFile testFile = inputFile(projectDir, "tests/unit/test_only.py", "def test_something():\n    pass\n", InputFile.Type.TEST);
    var localIndexer = buildIndexer(projectDir, packageInit, module, testFile);

    localIndexer.process(event(ModuleFileEvent.Type.DELETED, testFile));

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.module.original")).isNotNull();
    assertThat(localIndexer.getFileWithId(testFile.wrappedFile().absolutePath())).isNull();
  }

  @Test
  void deleted_file_with_another_root_contributor_skips_root_resolution(@TempDir Path projectDir) throws IOException {
    PythonInputFile first = inputFile(projectDir, "scripts/first.py", "def first():\n    pass\n");
    PythonInputFile second = inputFile(projectDir, "scripts/second.py", "def second():\n    pass\n");
    PythonInputFile third = inputFile(projectDir, "scripts/third.py", "def third():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, first, second, third));

    localIndexer.process(event(ModuleFileEvent.Type.DELETED, first));
    clearInvocations(localIndexer);
    localIndexer.process(event(ModuleFileEvent.Type.DELETED, second));

    verify(localIndexer, never()).refreshPackageRoots();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("first.first")).isNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("second.second")).isNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("third.third")).isNotNull();
  }

  @Test
  void modified_main_file_reclassified_as_test_is_removed_from_project_index(@TempDir Path projectDir) throws IOException {
    PythonInputFile mainFile = inputFile(projectDir, "module.py", "def original():\n    pass\n");
    var localIndexer = buildIndexer(projectDir, mainFile);

    PythonInputFile testFile = inputFile(projectDir, "module.py", "def test_something():\n    pass\n", InputFile.Type.TEST);
    localIndexer.process(event(ModuleFileEvent.Type.MODIFIED, testFile));

    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.original")).isNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.test_something")).isNull();
    assertThat(localIndexer.getFileWithId(testFile.wrappedFile().absolutePath())).isNull();
  }

  @Test
  void modified_test_file_reclassified_as_main_is_added_to_project_index(@TempDir Path projectDir) throws IOException {
    PythonInputFile testFile = inputFile(projectDir, "module.py", "def test_something():\n    pass\n", InputFile.Type.TEST);
    var localIndexer = buildIndexer(projectDir, testFile);

    PythonInputFile mainFile = inputFile(projectDir, "module.py", "def production():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.MODIFIED, mainFile));

    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.production")).isNotNull();
    assertThat(localIndexer.getFileWithId(mainFile.wrappedFile().absolutePath())).isSameAs(mainFile.wrappedFile());
  }

  @Test
  void deleted_file_removes_legacy_root_and_rebuilds_remaining_modules(@TempDir Path projectDir) throws IOException {
    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def original():\n    pass\n");
    PythonInputFile script = inputFile(projectDir, "scripts/tool.py", "def run():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, packageInit, module, script));

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.resolve("scripts").toString(), projectDir.toString());

    localIndexer.process(event(ModuleFileEvent.Type.DELETED, script));

    verify(localIndexer).refreshPackageRoots();
    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("tool.run")).isNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.module.original")).isNotNull();
  }

  @Test
  void deleted_file_with_removed_conventional_directory_refreshes_roots(@TempDir Path projectDir) throws IOException {
    Files.writeString(projectDir.resolve("pyproject.toml"), "[project]\nname = 'sample'\n");
    PythonInputFile script = inputFile(projectDir, "src/scripts/tool.py", "def run():\n    pass\n");
    var localIndexer = spy(buildIndexer(projectDir, script));
    Files.delete(script.wrappedFile().path());
    Files.delete(projectDir.resolve("src/scripts"));
    Files.delete(projectDir.resolve("src"));

    localIndexer.process(event(ModuleFileEvent.Type.DELETED, script));

    verify(localIndexer).refreshPackageRoots();
    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.toString());
  }

  @Test
  void created_src_folder_switches_from_base_dir_to_conventional_roots(@TempDir Path projectDir) throws IOException {
    Files.writeString(projectDir.resolve("pyproject.toml"), "[project]\nname = 'sample'\n");
    PythonInputFile main = inputFile(projectDir, "main.py", "def original():\n    pass\n");
    var localIndexer = buildIndexer(projectDir, main);

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.toString());

    PythonInputFile source = inputFile(projectDir, "src/tool.py", "def run():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, source));

    assertThat(localIndexer.packageRoots()).containsExactly(projectDir.resolve("src").toString(), projectDir.toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("tool.run")).isNotNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("src.tool.run")).isNull();
  }

  @Test
  void rebuild_continues_after_invalid_python_and_keeps_it_for_root_resolution(@TempDir Path projectDir) throws IOException {
    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def original():\n    pass\n");
    PythonInputFile invalid = inputFile(projectDir, "broken/invalid.py", "foo(");
    var localIndexer = buildIndexer(projectDir, packageInit, module, invalid);

    assertThat(localIndexer.getFileWithId(invalid.wrappedFile().absolutePath())).isNull();

    PythonInputFile script = inputFile(projectDir, "scripts/tool.py", "def run():\n    pass\n");
    assertDoesNotThrow(() -> localIndexer.process(event(ModuleFileEvent.Type.CREATED, script)));

    assertThat(localIndexer.packageRoots()).containsExactly(
      projectDir.resolve("broken").toString(),
      projectDir.resolve("scripts").toString(),
      projectDir.toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.module.original")).isNotNull();
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("tool.run")).isNotNull();
    assertThat(localIndexer.getFileWithId(invalid.wrappedFile().absolutePath())).isNull();
  }

  @Test
  void modified_file_can_be_temporarily_invalid_and_recover(@TempDir Path projectDir) throws IOException {
    PythonInputFile original = inputFile(projectDir, "module.py", "def original():\n    pass\n");
    var localIndexer = buildIndexer(projectDir, original);

    PythonInputFile invalid = inputFile(projectDir, "module.py", "foo(");
    assertDoesNotThrow(() -> localIndexer.process(event(ModuleFileEvent.Type.MODIFIED, invalid)));

    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.original")).isNull();
    assertThat(localIndexer.getFileWithId(invalid.wrappedFile().absolutePath())).isNull();

    PythonInputFile recovered = inputFile(projectDir, "module.py", "def recovered():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.MODIFIED, recovered));

    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("module.recovered")).isNotNull();
    assertThat(localIndexer.getFileWithId(recovered.wrappedFile().absolutePath())).isSameAs(recovered.wrappedFile());
  }

  @Test
  void rebuild_keeps_unreadable_file_for_root_resolution(@TempDir Path projectDir) throws IOException {
    PythonInputFile packageInit = inputFile(projectDir, "pkg/__init__.py", "");
    PythonInputFile module = inputFile(projectDir, "pkg/module.py", "def original():\n    pass\n");
    PythonInputFile readableInput = inputFile(projectDir, "unreadable/module.py", "def ignored():\n    pass\n");
    InputFile unreadableFile = spy(readableInput.wrappedFile());
    when(unreadableFile.contents()).thenThrow(new IOException("not readable"));
    PythonInputFile unreadableInput = new PythonInputFileImpl(unreadableFile);
    var localIndexer = buildIndexer(projectDir, packageInit, module, unreadableInput);

    PythonInputFile script = inputFile(projectDir, "scripts/tool.py", "def run():\n    pass\n");
    assertDoesNotThrow(() -> localIndexer.process(event(ModuleFileEvent.Type.CREATED, script)));

    assertThat(localIndexer.packageRoots()).containsExactly(
      projectDir.resolve("scripts").toString(),
      projectDir.resolve("unreadable").toString(),
      projectDir.toString());
    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("pkg.module.original")).isNotNull();
    assertThat(localIndexer.getFileWithId(unreadableFile.absolutePath())).isNull();
  }

  @Test
  void project_over_line_limit_does_not_partially_activate_after_file_event(@TempDir Path projectDir) throws IOException {
    PythonInputFile module = inputFile(projectDir, "module.py", "def original():\n    pass\n");
    var localContext = SensorContextTester.create(projectDir.toFile());
    localContext.setSettings(new MapSettings().setProperty("sonar.python.sonarlint.indexing.maxlines", 1));
    localContext.fileSystem().add(module.wrappedFile());
    var localIndexer = new SonarLintPythonIndexer(
      new TestModuleFileSystem(new ArrayList<>(List.of(module))), new ProjectConfigurationBuilder());
    localIndexer.buildOnce(localContext);

    PythonInputFile added = inputFile(projectDir, "scripts/tool.py", "def run():\n    pass\n");
    localIndexer.process(event(ModuleFileEvent.Type.CREATED, added));

    assertThat(localIndexer.projectLevelSymbolTable().getSymbol("tool.run")).isNull();
    assertThat(localIndexer.getFileWithId(added.wrappedFile().absolutePath())).isNull();
  }

  @Test
  void test_indexer_non_python_file() {
    testNonPythonFile("txt");
    testNonPythonFile(null);
  }

  @Test
  void test_indexer_added_file_updates_type_table() {
    
    // Verify that the "added" module type is initially unknown
    PythonType addedModuleTypeBefore = pythonIndexer.projectLevelTypeTable().getModuleType(List.of("added_type"));
    assertThat(addedModuleTypeBefore).isInstanceOf(UnknownType.class);

    // Create and add a new file
    PythonInputFile file = inputFile("added_type.py");
    ModuleFileEvent moduleFileEvent = mock(ModuleFileEvent.class);
    when(moduleFileEvent.getType()).thenReturn(ModuleFileEvent.Type.CREATED);
    when(moduleFileEvent.getTarget()).thenReturn(file.wrappedFile());
    pythonIndexer.process(moduleFileEvent);

    PythonType addedModuleTypeAfter = pythonIndexer.projectLevelTypeTable().getModuleType(List.of("added_type"));
    assertThat(addedModuleTypeAfter).isInstanceOf(ModuleType.class);
    assertThat(addedModuleTypeAfter.toString()).contains("added_type");

    PythonType addedFunction = pythonIndexer.projectLevelTypeTable().getType("added_type.A.foo");
    assertThat(addedFunction).isInstanceOf(FunctionType.class);
    assertThat(addedFunction.toString()).contains("FunctionType[foo]");

  }

  @Test
  void test_indexer_removed_file_updates_type_table() {
    // Create and add a new file
    PythonInputFile file = inputFile("removed_type.py");
    ModuleFileEvent moduleFileEvent = mock(ModuleFileEvent.class);
    when(moduleFileEvent.getType()).thenReturn(ModuleFileEvent.Type.CREATED);
    when(moduleFileEvent.getTarget()).thenReturn(file.wrappedFile());
    pythonIndexer.process(moduleFileEvent);

    assertThat(projectLevelSymbolTable.getSymbolsFromModule("removed_type")).isNotNull();
    PythonType addedModule = pythonIndexer.projectLevelTypeTable().getModuleType(List.of("removed_type"));
    assertThat(addedModule).isInstanceOf(ModuleType.class);
    assertThat(addedModule.toString()).contains("removed_type");

    PythonType functionBar = pythonIndexer.projectLevelTypeTable().getType("removed_type.B.bar");
    assertThat(functionBar).isInstanceOf(FunctionType.class);
    assertThat(functionBar.toString()).contains("FunctionType[bar]");

    when(moduleFileEvent.getType()).thenReturn(ModuleFileEvent.Type.DELETED);
    when(moduleFileEvent.getTarget()).thenReturn(file.wrappedFile());
    pythonIndexer.process(moduleFileEvent);
    
    PythonType removedModule = pythonIndexer.projectLevelTypeTable().getModuleType(List.of("removed_type"));
    assertThat(removedModule).isInstanceOf(UnknownType.class);
    PythonType removedClass = pythonIndexer.projectLevelTypeTable().getType("removed_type.B");
    assertThat(removedClass).isInstanceOf(UnknownType.class);

    // Modifying the file here would act a simply recreating it
    when(moduleFileEvent.getType()).thenReturn(ModuleFileEvent.Type.MODIFIED);
    when(moduleFileEvent.getTarget()).thenReturn(file.wrappedFile());
    pythonIndexer.process(moduleFileEvent);

    PythonType reAddedFun = pythonIndexer.projectLevelTypeTable().getType("removed_type.B.bar");
    assertThat(reAddedFun).isInstanceOf(FunctionType.class);
    assertThat(reAddedFun.toString()).contains("FunctionType[bar]");
  }

  @Test
  void test_sonarlint_cache() throws IOException {
    PythonIndexer indexer = new SonarLintPythonIndexer(moduleFileSystem, new ProjectConfigurationBuilder());
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
    assertDoesNotThrow(() -> pythonIndexer.process(moduleFileEvent), "Non Python files should not be parsed.");
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

  private static PythonInputFile inputFile(Path projectDir, String relativePath, String content) throws IOException {
    return inputFile(projectDir, relativePath, content, InputFile.Type.MAIN);
  }

  private static PythonInputFile inputFile(Path projectDir, String relativePath, String content, InputFile.Type type) throws IOException {
    Path path = projectDir.resolve(relativePath);
    Files.createDirectories(path.getParent());
    Files.writeString(path, content);
    return new PythonInputFileImpl(TestInputFileBuilder.create("moduleKey", relativePath)
      .setModuleBaseDir(projectDir)
      .setCharset(StandardCharsets.UTF_8)
      .setType(type)
      .setLanguage(Python.KEY)
      .initMetadata(content)
      .build());
  }

  private static ModuleFileEvent event(ModuleFileEvent.Type type, PythonInputFile inputFile) {
    return event(type, inputFile.wrappedFile());
  }

  private static ModuleFileEvent event(ModuleFileEvent.Type type, InputFile inputFile) {
    ModuleFileEvent event = mock(ModuleFileEvent.class);
    when(event.getType()).thenReturn(type);
    when(event.getTarget()).thenReturn(inputFile);
    return event;
  }

  private static SonarLintPythonIndexer buildIndexer(Path projectDir, PythonInputFile... files) {
    var localContext = SensorContextTester.create(projectDir.toFile());
    Arrays.stream(files).map(PythonInputFile::wrappedFile).forEach(localContext.fileSystem()::add);
    var localIndexer = new SonarLintPythonIndexer(
      new TestModuleFileSystem(new ArrayList<>(Arrays.asList(files))), new ProjectConfigurationBuilder());
    localIndexer.buildOnce(localContext);
    return localIndexer;
  }
}
