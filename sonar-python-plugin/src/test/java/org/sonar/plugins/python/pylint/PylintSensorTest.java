/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.plugins.python.pylint;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.batch.rule.internal.NewActiveRule;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.config.internal.ConfigurationBridge;
import org.sonar.api.config.internal.MapSettings;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.utils.log.LogTester;
import org.sonar.api.utils.log.LoggerLevel;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.TestUtils;

import static java.util.Arrays.asList;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class PylintSensorTest {

  private static final String FILE1_PATH = "file1.py";
  private static final File moduleBaseDir = new File("src/test/resources/org/sonar/plugins/python/pylint").getAbsoluteFile();
  public static final String C0103_RULE_KEY = "C0103";

  @Rule
  public LogTester logTester = new LogTester();
  @Rule
  public TemporaryFolder tmpDir = new TemporaryFolder();

  private File workDir = null;
  private PylintConfiguration conf;

  @Before
  public void setup() throws Exception {
    conf = mock(PylintConfiguration.class);
    if (workDir == null) {
      setupWorkDir();
    }
  }

  @Test
  public void sensor_descriptor() {
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    new PylintSensor(conf, new ConfigurationBridge(new MapSettings())).describe(descriptor);
    assertThat(descriptor.name()).isEqualTo("PylintSensor");
    assertThat(descriptor.languages()).containsOnly("py");
    assertThat(descriptor.type()).isEqualTo(InputFile.Type.MAIN);
    assertThat(descriptor.ruleRepositories()).containsExactly(PylintRuleRepository.REPOSITORY_KEY);
  }

  @Test
  public void shouldExecuteOnlyWhenNecessary() {
    assertThat(shouldExecute(null)).isTrue();
    assertThat(shouldExecute("result.txt")).isFalse();
  }

  @Test
  public void testWhenNoPylint() {
    SensorContextTester context = SensorContextTester.create(workDir);
    context.fileSystem().setWorkDir(workDir.toPath());
    when(conf.getPylintPath()).thenReturn("[---/this/should/definitely/not/exist---]");

    PylintSensor sensor = new PylintSensor(conf, new ConfigurationBridge(new MapSettings()));
    sensor.execute(context);
    assertThat(logTester.logs(LoggerLevel.WARN)).contains("Unable to use pylint for analysis. Error:");
  }

  @Test
  public void testWithFakePylint() throws IOException {
    SensorContextTester context = SensorContextTester.create(workDir);
    context.fileSystem().setWorkDir(workDir.toPath());

    createInputFile(workDir, context, FILE1_PATH);

    context.setActiveRules(
      new ActiveRulesBuilder()
        .addRule(new NewActiveRule.Builder()
          .setRuleKey(RuleKey.of(PylintRuleRepository.REPOSITORY_KEY, C0103_RULE_KEY))
          .setName("Invalid name")
          .build())
        .build());

    Issue issue1 = new Issue(FILE1_PATH, 1, C0103_RULE_KEY, "name1", "desc1");
    Issue issue2 = new Issue(FILE1_PATH, 2, "C0111", "name2", "desc2");
    Issue issue3 = new Issue(FILE1_PATH, 3, "C9999", "name3", "desc3");

    MapSettings settings = new MapSettings();
    settings.setProperty(PylintImportSensor.REPORT_PATH_KEY, "report-is-set");
    PylintSensor sensor = spy(new PylintSensor(conf, new ConfigurationBridge(settings)));
    PylintIssuesAnalyzer analyzer = mock(PylintIssuesAnalyzer.class);
    when(analyzer.analyze(any(), any(), any())).thenReturn(asList(issue1, issue2, issue3));
    doReturn(analyzer).when(sensor).createAnalyzer(any(), any());

    sensor.execute(context);

    // sensor was not executed as the 'sonar.python.pylint.reportPath' is set
    assertThat(context.allIssues()).hasSize(0);

    PylintImportSensor.clearLoggedWarnings();
    settings.clear();
    sensor.execute(context);

    assertThat(context.allIssues()).hasSize(1);
    assertThat(context.allIssues().iterator().next().ruleKey().rule()).isEqualTo(C0103_RULE_KEY);
    assertThat(logTester.logs(LoggerLevel.WARN)).contains("Pylint rule 'C9999' is unknown in Sonar");
    assertThat(logTester.logs(LoggerLevel.WARN)).doesNotContain("Pylint rule 'C0111' is unknown in Sonar");
  }

  @Test
  public void testErrorOnFileContinueAnalysis() throws IOException {
    SensorContextTester context = SensorContextTester.create(workDir);
    context.fileSystem().setWorkDir(workDir.toPath());

    createInputFile(workDir, context, FILE1_PATH);
    createInputFile(workDir, context, "file2.py");

    PylintSensor sensor = spy(new PylintSensor(conf, new ConfigurationBridge(new MapSettings())));
    PylintIssuesAnalyzer analyzer = mock(PylintIssuesAnalyzer.class);
    when(analyzer.analyze(any(), any(), any())).thenThrow(RuntimeException.class);
    doReturn(analyzer).when(sensor).createAnalyzer(any(), any());

    sensor.execute(context);
    assertThat(logTester.logs(LoggerLevel.WARN)).contains("Cannot analyse file 'file1.py', the following exception occurred:");
    verify(analyzer, times(2)).analyze(any(), any(), any());
  }

  private static void createInputFile(File baseDir, SensorContextTester context, String filePath) {
    File file = new File(baseDir, filePath);
    DefaultInputFile inputFile = TestInputFileBuilder.create("", filePath)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(file, StandardCharsets.UTF_8))
      .build();
    context.fileSystem().add(inputFile);
  }

  private boolean shouldExecute(@Nullable String pylintReportPath) {
    MapSettings settings = new MapSettings();
    if (pylintReportPath != null) {
      settings.setProperty(PylintImportSensor.REPORT_PATH_KEY, pylintReportPath);
    }
    PylintSensor sensor = new PylintSensor(conf, new ConfigurationBridge(settings));
    return sensor.shouldExecute();
  }

  private void setupWorkDir() throws Exception {
    workDir = tmpDir.newFolder("python-pylint");

    Path file1SourcePath = new File(moduleBaseDir, "src/file1.py").toPath();
    Path file1TargetPath = new File(workDir, FILE1_PATH).toPath();
    Path file2SourcePath = new File(moduleBaseDir, "src/file2.py").toPath();
    Path file2TargetPath = new File(workDir, "file2.py").toPath();

    Files.copy(file1SourcePath, file1TargetPath);
    Files.copy(file2SourcePath, file2TargetPath);
  }

}
