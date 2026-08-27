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
package org.sonar.plugins.python;

import com.sonarsource.scanner.engine.sensor.test.fixtures.SensorContextTester;
import com.sonarsource.scanner.engine.sensor.test.fixtures.TestInputFileBuilder;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.plugins.python.api.PythonCustomRuleRepositoryWrapper;
import org.sonar.plugins.python.api.SonarLintCacheWrapper;
import org.sonar.plugins.python.architecture.ArchitectureCallbackWrapper;
import org.sonar.plugins.python.editions.RepositoryInfoProviderWrapper;
import org.sonar.plugins.python.indexer.ProjectLevelSymbolTableWrapper;
import org.sonar.plugins.python.indexer.PythonIndexerWrapper;
import org.sonar.plugins.python.nosonar.NoSonarLineInfoCollector;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;
import org.sonar.python.project.config.ProjectConfigurationBuilder;
import org.sonar.python.types.TypeShed;
import org.sonar.scanner.plugin.api.impl.rule.ActiveRulesBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Integration test verifying that {@link PythonSensor} populates {@link ProjectLevelSymbolTableWrapper}
 * after the analysis runs, making the symbol table accessible to subsequent sensors.
 */
class ProjectLevelSymbolTableWrapperTest {

  @TempDir
  File tempDir;

  private SensorContextTester context;
  private ProjectLevelSymbolTableWrapper symbolTableWrapper;
  private PythonSensor sensor;

  @BeforeEach
  void setUp() {
    context = SensorContextTester.create(tempDir);
    context.fileSystem().setWorkDir(tempDir.toPath().resolve("work"));
    TypeShed.resetBuiltinSymbols();

    symbolTableWrapper = new ProjectLevelSymbolTableWrapper();

    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(any(InputFile.class))).thenReturn(fileLinesContext);

    sensor = new PythonSensor(
      fileLinesContextFactory,
      new CheckFactory(new ActiveRulesBuilder().build()),
      mock(NoSonarFilter.class),
      new PythonCustomRuleRepositoryWrapper(null),
      new PythonIndexerWrapper(null),  // null → SonarQubePythonIndexer created internally
      new SonarLintCacheWrapper(),
      mock(AnalysisWarningsWrapper.class),
      new RepositoryInfoProviderWrapper(),
      new ArchitectureCallbackWrapper(),
      new NoSonarLineInfoCollector(),
      new ProjectConfigurationBuilder(),
      symbolTableWrapper
    );
  }

  @Test
  void symbolTableWrapperIsEmptyBeforeSensorRuns() {
    assertThat(symbolTableWrapper.symbolTable().globalDescriptorsByModuleName()).isEmpty();
  }

  @Test
  void symbolTableWrapperIsPopulatedAfterSensorRuns() throws IOException {
    // Create a simple Python file with a function definition so the symbol table has content
    File initFile = new File(tempDir, "mypackage/__init__.py");
    initFile.getParentFile().mkdirs();
    Files.writeString(initFile.toPath(), "");

    File pyFile = new File(tempDir, "mypackage/foo.py");
    Files.writeString(pyFile.toPath(), "def my_function(): pass\n");

    context.fileSystem().add(
      TestInputFileBuilder.create("moduleKey", tempDir, pyFile)
        .setLanguage("py")
        .setType(InputFile.Type.MAIN)
        .setCharset(StandardCharsets.UTF_8)
        .build());

    sensor.execute(context);

    // The wrapper should now hold the populated symbol table
    assertThat(symbolTableWrapper.symbolTable().globalDescriptorsByModuleName())
      .isNotEmpty()
      .containsKey("mypackage.foo");
  }

  @Test
  void symbolTableWrapperHoldsDescriptorsForAllAnalysedModules() throws IOException {
    File initFile = new File(tempDir, "pkg/__init__.py");
    initFile.getParentFile().mkdirs();
    Files.writeString(initFile.toPath(), "");

    File fileA = new File(tempDir, "pkg/mod_a.py");
    Files.writeString(fileA.toPath(), "class A: pass\n");

    File fileB = new File(tempDir, "pkg/mod_b.py");
    Files.writeString(fileB.toPath(), "class B: pass\n");

    context.fileSystem().add(
      TestInputFileBuilder.create("moduleKey", tempDir, fileA)
        .setLanguage("py").setType(InputFile.Type.MAIN).setCharset(StandardCharsets.UTF_8).build());
    context.fileSystem().add(
      TestInputFileBuilder.create("moduleKey", tempDir, fileB)
        .setLanguage("py").setType(InputFile.Type.MAIN).setCharset(StandardCharsets.UTF_8).build());

    sensor.execute(context);

    assertThat(symbolTableWrapper.symbolTable().globalDescriptorsByModuleName())
      .containsKeys("pkg.mod_a", "pkg.mod_b");
  }
}
