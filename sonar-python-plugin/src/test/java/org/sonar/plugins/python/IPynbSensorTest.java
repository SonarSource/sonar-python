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
package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.mockito.Mockito;
import org.slf4j.event.Level;
import org.sonar.api.SonarRuntime;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.batch.rule.internal.NewActiveRule;
import org.sonar.api.batch.sensor.cpd.internal.TokensLine;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.config.internal.MapSettings;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.api.utils.Version;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.plugins.python.indexer.SonarLintPythonIndexer;
import org.sonar.plugins.python.indexer.TestModuleFileSystem;
import org.sonar.python.checks.CheckList;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class IPynbSensorTest {

  private static final Version SONARLINT_DETECTABLE_VERSION = Version.create(9, 9);

  static final SonarRuntime SONARLINT_RUNTIME = SonarRuntimeImpl.forSonarLint(SONARLINT_DETECTABLE_VERSION);

  private static final String FILE_1 = "file1.ipynb";
  private static final String NOTEBOOK_FILE = "notebook.ipynb";

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/ipynb/").getAbsoluteFile();

  private SensorContextTester context;

  private ActiveRules activeRules;

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @BeforeEach
  void init() throws IOException {
    context = SensorContextTester.create(baseDir);
    var workDir = Files.createTempDirectory("workDir");
    context.fileSystem().setWorkDir(workDir);
  }

  @Test
  void sensor_descriptor() {
    activeRules = new ActiveRulesBuilder().build();
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    sensor().describe(descriptor);

    assertThat(descriptor.name()).isEqualTo("IPython Notebooks Sensor");
    assertThat(descriptor.languages()).containsOnly("ipynb");
    assertThat(descriptor.type()).isNull();
  }

  @Test
  void test_execute_on_sonarlint() {
    context.setRuntime(SONARLINT_RUNTIME);

    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.IPYTHON_REPOSITORY_KEY, "PrintStatementUsage"))
        .setName("Print Statement Usage")
        .build())
      .build();

    PythonInputFile inputFile = inputFile(FILE_1);

    PythonIndexer pythonIndexer = pythonIndexer(List.of(inputFile));
    sensor(pythonIndexer).execute(context);

    String key = "moduleKey:file1.ipynb";
    assertThat(context.measure(key, CoreMetrics.NCLOC)).isNull();
    assertThat(context.allIssues()).hasSize(1);
    assertThat(context.highlightingTypeAt(key, 15, 2)).isEmpty();
    assertThat(context.allAnalysisErrors()).isEmpty();

    assertThat(PythonScanner.getWorkingDirectory(context)).isNull();
  }

  private IPynbSensor sensor() {
    return sensor(pythonIndexer(Collections.emptyList()));
  }

  private IPynbSensor sensor(PythonIndexer indexer) {
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(InputFile.class))).thenReturn(fileLinesContext);
    CheckFactory checkFactory = new CheckFactory(activeRules);
    return new IPynbSensor(fileLinesContextFactory, checkFactory, mock(NoSonarFilter.class), indexer);
  }

  private PythonInputFile inputFile(String name) {
    PythonInputFile inputFile = createInputFile(name);
    context.fileSystem().add(inputFile.wrappedFile());
    return inputFile;
  }

  private PythonInputFile createInputFile(String name) {
    return new PythonInputFileImpl(TestInputFileBuilder.create("moduleKey", name)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(UTF_8)
      .setType(InputFile.Type.MAIN)
      .setLanguage(IPynb.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, name), UTF_8))
      .setStatus(InputFile.Status.ADDED)
      .build());
  }

  private SonarLintPythonIndexer pythonIndexer(List<PythonInputFile> files) {
    return new SonarLintPythonIndexer(new TestModuleFileSystem(files));
  }

  @Test
  void test_python_version_parameter() {
    context.setRuntime(SONARLINT_RUNTIME);

    PythonInputFile inputFile = inputFile(FILE_1);
    activeRules = new ActiveRulesBuilder().build();
    context.setSettings(new MapSettings().setProperty("sonar.python.version", "3.8"));
    PythonIndexer pythonIndexer = pythonIndexer(List.of(inputFile));

    sensor(pythonIndexer).execute(context);

    assertThat(ProjectPythonVersion.currentVersions()).containsExactly(PythonVersionUtils.Version.V_38);
  }

  private IPynbSensor notebookSensor() {
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(InputFile.class))).thenReturn(fileLinesContext);
    CheckFactory checkFactory = new CheckFactory(activeRules);
    return new IPynbSensor(fileLinesContextFactory, checkFactory, mock(NoSonarFilter.class));
  }

  @Test
  void test_notebook_sensor_error_should_throw_if_fail_fast() {
    context.settings().setProperty("sonar.internal.analysis.failFast", true);
    inputFile(FILE_1);
    activeRules = new ActiveRulesBuilder().build();
    IPynbSensor sensor = notebookSensor();
    Throwable throwable = assertThrows(IllegalStateException.class, () -> sensor.execute(context));
    assertThat(throwable.getClass()).isEqualTo(IllegalStateException.class);
    assertThat(throwable.getMessage()).isEqualTo("Exception when parsing file1.ipynb");
  }

  @Test
  void test_notebook_sensor_should_not_throw_on_test_file() {
    context.settings().setProperty("sonar.internal.analysis.failFast", true);
    PythonInputFile testFile = new PythonInputFileImpl(TestInputFileBuilder.create("moduleKey", FILE_1)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(UTF_8)
      .setType(InputFile.Type.TEST)
      .setLanguage(IPynb.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, FILE_1), UTF_8))
      .setStatus(InputFile.Status.ADDED)
      .build());
    context.fileSystem().add(testFile.wrappedFile());
    activeRules = new ActiveRulesBuilder().build();
    assertDoesNotThrow(() -> notebookSensor().execute(context));
  }

  @Test
  void test_notebook_sensor_cannot_parse_file() {
    inputFile(FILE_1);
    activeRules = new ActiveRulesBuilder().build();
    assertDoesNotThrow(() -> notebookSensor().execute(context));
  }

  @Test
  void test_notebook_sensor_is_excuted_on_json_file() {
    inputFile(NOTEBOOK_FILE);
    activeRules = new ActiveRulesBuilder().build();
    assertDoesNotThrow(() -> notebookSensor().execute(context));
  }

  @Test
  void test_notebook_sensor_does_not_execute_cpd_measures() {
    inputFile(NOTEBOOK_FILE);
    activeRules = new ActiveRulesBuilder().build();
    notebookSensor().execute(context);
    List<TokensLine> tokensLines = context.cpdTokens("moduleKey:notebook.ipynb");
    assertThat(tokensLines)
      .isNull();
  }

  @Test
  void test_notebook_sensor_parse_error_on_valid_line(){
    inputFile("notebook_parse_error.ipynb");
    activeRules = new ActiveRulesBuilder().build();
    var sensor = notebookSensor();
    sensor.execute(context);
    var logs = String.join("", logTester.logs());
    assertThat(logs).contains("Unable to parse file: notebook_parse_error.ipynbParse error at line 1");
  }
}
