/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.api.SonarQubeSide;
import org.sonar.api.SonarRuntime;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.batch.sensor.error.AnalysisError;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.batch.sensor.issue.Issue;
import org.sonar.api.batch.sensor.issue.IssueLocation;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.utils.Version;
import org.sonar.api.utils.log.LogTester;
import org.sonar.plugins.python.coverage.PythonCoverageSensor;
import org.sonar.python.checks.CheckList;
import org.sonar.python.checks.ParsingErrorCheck;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class PythonSquidSensorTest {

  private static final Version SONARLINT_DETECTABLE_VERSION = Version.create(6, 0);

  private static final SonarRuntime SONARLINT_RUNTIME = SonarRuntimeImpl.forSonarLint(SONARLINT_DETECTABLE_VERSION);

  private static final SonarRuntime NOSONARLINT_RUNTIME = SonarRuntimeImpl.forSonarQube(SONARLINT_DETECTABLE_VERSION, SonarQubeSide.SERVER);

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/squid-sensor").getAbsoluteFile();

  private SensorContextTester context;

  private ActiveRules activeRules;

  @org.junit.Rule
  public LogTester logTester = new LogTester();

  @Before
  public void init() {
    context = SensorContextTester.create(baseDir);
    context.settings().setProperty(PythonCoverageSensor.OVERALL_REPORT_PATH_KEY, "coverage.xml");
  }

  @Test
  public void sensor_descriptor() {
    activeRules = (new ActiveRulesBuilder()).build();
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    sensor().describe(descriptor);

    assertThat(descriptor.name()).isEqualTo("Python Squid Sensor");
    assertThat(descriptor.languages()).containsOnly("py");
    assertThat(descriptor.type()).isEqualTo(Type.MAIN);
  }

  @Test
  public void test_execute_on_sonarqube() {
    // with SonarQube configuration, coverage is activated
    test_execute(NOSONARLINT_RUNTIME, 10);
  }

  @Test
  public void test_execute_on_sonarlint() {
    // with SonarLint configuration, coverage is not activated
    test_execute(SONARLINT_RUNTIME, null);
  }

  private void test_execute(SonarRuntime runtime, Integer expectedNumberOfLineHits) {
    context.setRuntime(runtime);

    activeRules = (new ActiveRulesBuilder())
      .create(RuleKey.of(CheckList.REPOSITORY_KEY, "PrintStatementUsage"))
      .setName("Print Statement Usage")
      .activate()
      .build();

    inputFile("file1.py");

    sensor().execute(context);

    String key = "moduleKey:file1.py";
    assertThat(context.measure(key, CoreMetrics.NCLOC).value()).isEqualTo(22);
    assertThat(context.measure(key, CoreMetrics.STATEMENTS).value()).isEqualTo(20);
    assertThat(context.measure(key, CoreMetrics.FUNCTIONS).value()).isEqualTo(4);
    assertThat(context.measure(key, CoreMetrics.CLASSES).value()).isEqualTo(1);
    assertThat(context.measure(key, CoreMetrics.COMPLEXITY).value()).isEqualTo(4);
    assertThat(context.measure(key, CoreMetrics.COMMENT_LINES).value()).isEqualTo(8);

    assertThat(context.allIssues()).hasSize(1);

    String msg = "number of TypeOfText for the highlighting of keyword 'def'";
    assertThat(context.highlightingTypeAt(key, 15, 2)).as(msg).hasSize(1);

    assertThat(context.lineHits("moduleKey:file1.py", 1)).isEqualTo(expectedNumberOfLineHits);

    assertThat(context.allAnalysisErrors()).isEmpty();
  }

  @Test
  public void test_issues() {
    activeRules = (new ActiveRulesBuilder())
      .create(RuleKey.of(CheckList.REPOSITORY_KEY, "OneStatementPerLine"))
      .activate()
      .create(RuleKey.of(CheckList.REPOSITORY_KEY, "S134"))
      .activate()
      .create(RuleKey.of(CheckList.REPOSITORY_KEY, "FileComplexity"))
      .setParam("maximumFileComplexityThreshold", "2")
      .activate()
      .build();

    InputFile inputFile = inputFile("file2.py");
    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(3);
    Iterator<Issue> issuesIterator = context.allIssues().iterator();

    int checkedIssues = 0;

    while (issuesIterator.hasNext()) {
      Issue issue = issuesIterator.next();
      IssueLocation issueLocation = issue.primaryLocation();
      assertThat(issueLocation.inputComponent()).isEqualTo(inputFile);

      if (issue.ruleKey().rule().equals("S134")) {
        assertThat(issueLocation.message()).isEqualTo("Refactor this code to not nest more than 4 \"if\", \"for\", \"while\", \"try\" and \"with\" statements.");
        assertThat(issueLocation.textRange()).isEqualTo(inputFile.newRange(7, 16, 7, 18));
        assertThat(issue.flows()).hasSize(4);
        assertThat(issue.gap()).isNull();
        checkedIssues++;
      } else if (issue.ruleKey().rule().equals("OneStatementPerLine")) {
        assertThat(issueLocation.message()).isEqualTo("At most one statement is allowed per line, but 2 statements were found on this line.");
        assertThat(issueLocation.textRange()).isEqualTo(inputFile.newRange(1, 0, 1, 50));
        assertThat(issue.flows()).isEmpty();
        assertThat(issue.gap()).isNull();
        checkedIssues++;
      } else if (issue.ruleKey().rule().equals("FileComplexity")) {
        assertThat(issueLocation.message()).isEqualTo("File has a complexity of 5 which is greater than 2 authorized.");
        assertThat(issueLocation.textRange()).isNull();
        assertThat(issue.flows()).isEmpty();
        assertThat(issue.gap()).isEqualTo(3.0);
        checkedIssues++;
      } else {
        throw new IllegalStateException();
      }
    }

    assertThat(checkedIssues).isEqualTo(3);
  }

  @Test
  public void parse_error() throws Exception {
    inputFile("parse_error.py");
    activeRules = (new ActiveRulesBuilder())
      .create(RuleKey.of(CheckList.REPOSITORY_KEY, ParsingErrorCheck.CHECK_KEY))
      .activate()
      .build();
    sensor().execute(context);
    assertThat(context.allIssues()).hasSize(1);
    assertThat(String.join("\n", logTester.logs())).contains("Parse error at line 2");
    assertThat(context.allAnalysisErrors()).hasSize(1);
    AnalysisError analysisError = context.allAnalysisErrors().iterator().next();
    assertThat(analysisError.inputFile().filename()).isEqualTo("parse_error.py");
    assertThat(analysisError.location().line()).isEqualTo(2);
  }

  @Test
  public void cancelled_analysis() {
    InputFile inputFile = inputFile("file1.py");
    activeRules = (new ActiveRulesBuilder()).build();
    context.setCancelled(true);
    sensor().execute(context);
    assertThat(context.measure(inputFile.key(), CoreMetrics.NCLOC)).isNull();
    assertThat(context.allAnalysisErrors()).isEmpty();
  }

  private PythonSquidSensor sensor() {
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(InputFile.class))).thenReturn(fileLinesContext);
    CheckFactory checkFactory = new CheckFactory(activeRules);
    return new PythonSquidSensor(fileLinesContextFactory, checkFactory, new NoSonarFilter());
  }

  private InputFile inputFile(String name) {
    DefaultInputFile inputFile =  TestInputFileBuilder.create("moduleKey", name)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setType(Type.MAIN)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, name), StandardCharsets.UTF_8))
      .build();
    context.fileSystem().add(inputFile);
    return inputFile;
  }

}
