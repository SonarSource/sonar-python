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
package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.slf4j.event.Level;
import org.sonar.api.SonarEdition;
import org.sonar.api.SonarQubeSide;
import org.sonar.api.SonarRuntime;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.batch.rule.internal.NewActiveRule;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.batch.sensor.issue.Issue;
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
import org.sonarsource.sonarlint.core.commons.TextRange;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class IPynbSensorTest {

  private static final Version SONARLINT_DETECTABLE_VERSION = Version.create(9, 9);

  static final SonarRuntime SONARLINT_RUNTIME = SonarRuntimeImpl.forSonarLint(SONARLINT_DETECTABLE_VERSION);
  private static final SonarRuntime SONAR_RUNTIME = SonarRuntimeImpl.forSonarQube(Version.create(9, 9), SonarQubeSide.SERVER, SonarEdition.DEVELOPER);


  private static final String FILE_1 = "file1.ipynb";
  private static final String ACTUAL_NOTEBOOK = "actual_notebook.ipynb";

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/ipynb").getAbsoluteFile();

  private SensorContextTester context;

  private ActiveRules activeRules;

  @org.junit.Rule
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @BeforeEach
  void init() {
    context = SensorContextTester.create(baseDir);
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

    InputFile inputFile = inputFile(FILE_1);

    PythonIndexer pythonIndexer = pythonIndexer(List.of(inputFile));
    sensor(pythonIndexer).execute(context);

    String key = "moduleKey:file1.ipynb";
    assertThat(context.measure(key, CoreMetrics.NCLOC)).isNull();
    assertThat(context.allIssues()).hasSize(1);
    assertThat(context.highlightingTypeAt(key, 15, 2)).isEmpty();
    assertThat(context.allAnalysisErrors()).isEmpty();

    assertThat(PythonScanner.getWorkingDirectory(context)).isNull();
  }


  @Test
  void test_actual_notebook() throws IOException {
    context.setRuntime(SONAR_RUNTIME);

    Path workDir = Files.createTempDirectory("workDir");
    context.fileSystem().setWorkDir(workDir);

    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.IPYTHON_REPOSITORY_KEY, "PrintStatementUsage"))
        .setName("Print Statement Usage")
        .build())
      .build();

    InputFile inputFile = inputFile(ACTUAL_NOTEBOOK);

    PythonIndexer pythonIndexer = pythonIndexer(List.of(inputFile));
    sensor(pythonIndexer).execute(context);

    String key = "moduleKey:file1.ipynb";
    assertThat(context.measure(key, CoreMetrics.NCLOC)).isNull();
    assertThat(context.allIssues()).hasSize(1);
    assertThat(context.highlightingTypeAt(key, 15, 2)).isEmpty();
    assertThat(context.allAnalysisErrors()).isEmpty();

    assertThat(PythonScanner.getWorkingDirectory(context)).isNotNull();

    Collection<Issue> issues = context.allIssues();
    Issue issue = issues.iterator().next();
    assertThat(issue.primaryLocation().textRange()).isEqualTo(inputFile.newRange(19, 10, 19, 15));
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

  private InputFile inputFile(String name) {
    DefaultInputFile inputFile = createInputFile(name);
    context.fileSystem().add(inputFile);
    return inputFile;
  }

  private DefaultInputFile createInputFile(String name) {
    return TestInputFileBuilder.create("moduleKey", name)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(UTF_8)
      .setType(InputFile.Type.MAIN)
      .setLanguage(IPynb.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, name), UTF_8))
      .setStatus(InputFile.Status.ADDED)
      .build();
  }

  private SonarLintPythonIndexer pythonIndexer(List<InputFile> files) {
    return new SonarLintPythonIndexer(new TestModuleFileSystem(files));
  }
  @Test
  void test_python_version_parameter() {
    context.setRuntime(SONARLINT_RUNTIME);

    InputFile inputFile = inputFile(FILE_1);
    activeRules = new ActiveRulesBuilder().build();
    context.setSettings(new MapSettings().setProperty("sonar.python.version", "3.8"));
    PythonIndexer pythonIndexer = pythonIndexer(List.of(inputFile));

    sensor(pythonIndexer).execute(context);

    assertThat(ProjectPythonVersion.currentVersions()).containsExactly(PythonVersionUtils.Version.V_38);
  }
}
