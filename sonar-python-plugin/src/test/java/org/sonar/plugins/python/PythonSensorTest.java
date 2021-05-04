/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.api.SonarRuntime;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.fs.TextPointer;
import org.sonar.api.batch.fs.TextRange;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.DefaultTextPointer;
import org.sonar.api.batch.fs.internal.DefaultTextRange;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.batch.rule.internal.NewActiveRule;
import org.sonar.api.batch.sensor.error.AnalysisError;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.batch.sensor.issue.Issue;
import org.sonar.api.batch.sensor.issue.IssueLocation;
import org.sonar.api.config.internal.MapSettings;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.utils.Version;
import org.sonar.api.utils.log.LogTester;
import org.sonar.api.utils.log.LoggerLevel;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCustomRuleRepository;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.checks.CheckList;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

public class PythonSensorTest {

  private static final String FILE_1 = "file1.py";
  private static final String FILE_2 = "file2.py";
  private static final String ONE_STATEMENT_PER_LINE_RULE_KEY = "OneStatementPerLine";
  private static final String FILE_COMPLEXITY_RULE_KEY = "FileComplexity";

  private static final Version SONARLINT_DETECTABLE_VERSION = Version.create(6, 0);

  private static final SonarRuntime SONARLINT_RUNTIME = SonarRuntimeImpl.forSonarLint(SONARLINT_DETECTABLE_VERSION);

  private static final PythonCustomRuleRepository[] CUSTOM_RULES = {new PythonCustomRuleRepository() {
    @Override
    public String repositoryKey() {
      return "customKey";
    }

    @Override
    public List<Class> checkClasses() {
      return Collections.singletonList(MyCustomRule.class);
    }
  }};
  private static Path workDir;

  @Rule(
    key = "key",
    name = "name",
    description = "desc",
    tags = {"bug"})
  public static class MyCustomRule implements PythonCheck {
    @RuleProperty(
      key = "customParam",
      description = "Custom parameter",
      defaultValue = "value")
    public String customParam = "value";

    @Override
    public void scanFile(PythonVisitorContext visitorContext) {
      // do nothing
    }
  }

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/sensor").getAbsoluteFile();

  private SensorContextTester context;

  private ActiveRules activeRules;

  @org.junit.Rule
  public LogTester logTester = new LogTester();

  @Before
  public void init() throws IOException {
    context = SensorContextTester.create(baseDir);
    workDir = Files.createTempDirectory("workDir");
    context.fileSystem().setWorkDir(workDir);
  }

  @Test
  public void sensor_descriptor() {
    activeRules = new ActiveRulesBuilder().build();
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    sensor().describe(descriptor);

    assertThat(descriptor.name()).isEqualTo("Python Sensor");
    assertThat(descriptor.languages()).containsOnly("py");
    assertThat(descriptor.type()).isEqualTo(Type.MAIN);
  }

  @Test
  public void test_execute_on_sonarlint() {
    context.setRuntime(SONARLINT_RUNTIME);

    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "PrintStatementUsage"))
        .setName("Print Statement Usage")
        .build())
      .build();

    inputFile(FILE_1);

    sensor().execute(context);

    String key = "moduleKey:file1.py";
    assertThat(context.measure(key, CoreMetrics.NCLOC).value()).isEqualTo(22);
    assertThat(context.measure(key, CoreMetrics.STATEMENTS).value()).isEqualTo(22);
    assertThat(context.measure(key, CoreMetrics.FUNCTIONS).value()).isEqualTo(4);
    assertThat(context.measure(key, CoreMetrics.CLASSES).value()).isEqualTo(1);
    assertThat(context.measure(key, CoreMetrics.COMPLEXITY).value()).isEqualTo(5);
    assertThat(context.measure(key, CoreMetrics.COGNITIVE_COMPLEXITY).value()).isEqualTo(1);
    assertThat(context.measure(key, CoreMetrics.COMMENT_LINES).value()).isEqualTo(8);

    assertThat(context.allIssues()).hasSize(1);

    String msg = "number of TypeOfText for the highlighting of keyword 'def'";
    assertThat(context.highlightingTypeAt(key, 15, 2)).as(msg).hasSize(1);

    assertThat(context.allAnalysisErrors()).isEmpty();

    assertThat(PythonScanner.getWorkingDirectory(context)).isNull();
  }

  @Test
  public void test_symbol_visitor() {
    activeRules = new ActiveRulesBuilder().build();
    inputFile(FILE_2);
    inputFile("symbolVisitor.py");
    sensor().execute(context);

    String key = "moduleKey:file2.py";
    assertThat(context.referencesForSymbolAt(key, 1, 10)).isNull();
    verifyUsages(key, 3, 4, reference(4, 10, 4, 11),
      reference(6, 15, 6, 16), reference(7, 19, 7, 20));
    verifyUsages(key, 5, 12, reference(6, 19, 6, 20));

    key = "moduleKey:symbolVisitor.py";
    assertThat(context.referencesForSymbolAt(key, 1, 10)).isNull();
    verifyUsages(key, 1, 0, reference(29, 14, 29, 15), reference(30, 18, 30, 19));
    verifyUsages(key, 2, 0, reference(3, 6, 3, 7), reference(10, 4, 10, 5), reference(32, 1, 32, 2));
    verifyUsages(key, 5, 4, reference(6, 4, 6, 5), reference(7, 4, 7, 5),
      reference(8, 8, 8, 9), reference(13, 9, 13, 10));
    verifyUsages(key, 47, 5, reference(48, 14, 48, 17));
  }

  @Test
  public void test_issues() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, ONE_STATEMENT_PER_LINE_RULE_KEY))
        .build())
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S134"))
        .build())
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, FILE_COMPLEXITY_RULE_KEY))
        .setParam("maximumFileComplexityThreshold", "2")
        .build())
      .build();

    InputFile inputFile = inputFile(FILE_2);
    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(3);
    Iterator<Issue> issuesIterator = context.allIssues().iterator();

    int checkedIssues = 0;

    while (issuesIterator.hasNext()) {
      Issue issue = issuesIterator.next();
      IssueLocation issueLocation = issue.primaryLocation();
      assertThat(issueLocation.inputComponent()).isEqualTo(inputFile);

      switch (issue.ruleKey().rule()) {
        case "S134":
          assertThat(issueLocation.message()).isEqualTo("Refactor this code to not nest more than 4 \"if\", \"for\", \"while\", \"try\" and \"with\" statements.");
          assertThat(issueLocation.textRange()).isEqualTo(inputFile.newRange(7, 16, 7, 18));
          assertThat(issue.flows()).hasSize(4);
          assertThat(issue.gap()).isNull();
          checkedIssues++;
          break;
        case ONE_STATEMENT_PER_LINE_RULE_KEY:
          assertThat(issueLocation.message()).isEqualTo("At most one statement is allowed per line, but 2 statements were found on this line.");
          assertThat(issueLocation.textRange()).isEqualTo(inputFile.newRange(1, 0, 1, 50));
          assertThat(issue.flows()).isEmpty();
          assertThat(issue.gap()).isNull();
          checkedIssues++;
          break;
        case FILE_COMPLEXITY_RULE_KEY:
          assertThat(issueLocation.message()).isEqualTo("File has a complexity of 5 which is greater than 2 authorized.");
          assertThat(issueLocation.textRange()).isNull();
          assertThat(issue.flows()).isEmpty();
          assertThat(issue.gap()).isEqualTo(3.0);
          checkedIssues++;
          break;
        default:
          throw new IllegalStateException();
      }
    }

    assertThat(checkedIssues).isEqualTo(3);
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Starting global symbols computation");
    assertThat(logTester.logs(LoggerLevel.INFO)).contains("Starting rules execution");
    assertThat(logTester.logs(LoggerLevel.INFO).stream().filter(line -> line.equals("1 source file to be analyzed")).count()).isEqualTo(2);

    assertThat(PythonScanner.getWorkingDirectory(context)).isEqualTo(workDir.toFile());
  }

  @Test
  public void cross_files_secondary_locations() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S930"))
        .build())
      .build();

    InputFile mainFile = inputFile("main.py");
    InputFile modFile = inputFile("mod.py");
    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(1);
    Issue issue = context.allIssues().iterator().next();
    assertThat(issue.flows()).hasSize(1);
    Issue.Flow flow = issue.flows().get(0);
    assertThat(flow.locations()).hasSize(2);
    assertThat(flow.locations().get(0).inputComponent()).isEqualTo(mainFile);
    assertThat(flow.locations().get(1).inputComponent()).isEqualTo(modFile);
  }

  @Test
  public void loop_in_class_hierarchy() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S2710"))
        .build())
      .build();

    inputFile("modA.py");
    inputFile("modB.py");
    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(1);
  }

  @Test
  public void test_test_file_highlighting() throws IOException {
    activeRules = new ActiveRulesBuilder().build();

    DefaultInputFile inputFile1 = spy(TestInputFileBuilder.create("moduleKey", FILE_1)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setType(Type.TEST)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, FILE_1), StandardCharsets.UTF_8))
      .build());

    DefaultInputFile inputFile2 = spy(TestInputFileBuilder.create("moduleKey", FILE_2)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setType(Type.TEST)
      .setLanguage(Python.KEY)
      .build());

    DefaultInputFile inputFile3 = spy(TestInputFileBuilder.create("moduleKey", "parse_error.py")
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setType(Type.TEST)
      .setLanguage(Python.KEY)
      .build());

    context.fileSystem().add(inputFile1);
    context.fileSystem().add(inputFile2);
    context.fileSystem().add(inputFile3);
    sensor().execute(context);
    assertThat(logTester.logs()).contains("Starting test sources highlighting");
    assertThat(logTester.logs()).contains("Unable to parse file: parse_error.py");
    assertThat(logTester.logs()).contains("Unable to highlight test file: file2.py");
    assertThat(context.highlightingTypeAt(inputFile1.key(), 1, 2)).isNotEmpty();
  }

  @Test
  public void test_exception_does_not_fail_analysis() throws IOException {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, ONE_STATEMENT_PER_LINE_RULE_KEY))
        .build())
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, FILE_COMPLEXITY_RULE_KEY))
        .setParam("maximumFileComplexityThreshold", "2")
        .build())
      .build();

    DefaultInputFile inputFile = spy(TestInputFileBuilder.create("moduleKey", FILE_1)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(StandardCharsets.UTF_8)
      .setType(Type.MAIN)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, FILE_1), StandardCharsets.UTF_8))
      .build());
    when(inputFile.contents()).thenThrow(RuntimeException.class);

    context.fileSystem().add(inputFile);
    inputFile(FILE_2);

    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(2);
  }

  @Test
  public void test_exception_should_fail_analysis_if_configured_so() throws IOException {
    DefaultInputFile inputFile = spy(createInputFile(FILE_1));
    when(inputFile.contents()).thenThrow(FileNotFoundException.class);
    context.fileSystem().add(inputFile);

    activeRules = new ActiveRulesBuilder().build();
    context.setSettings(new MapSettings().setProperty("sonar.internal.analysis.failFast", "true"));

    assertThatThrownBy(() -> sensor().execute(context))
      .isInstanceOf(IllegalStateException.class)
      .hasCauseInstanceOf(FileNotFoundException.class);
  }

  @Test
  public void parse_error() {
    inputFile("parse_error.py");
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "ParsingError"))
        .build())
      .build();

    sensor().execute(context);
    assertThat(context.allIssues()).hasSize(1);
    String log = String.join("\n", logTester.logs());
    assertThat(log).contains("Parse error at line 2");
    assertThat(log).doesNotContain("java.lang.NullPointerException");
    assertThat(context.allAnalysisErrors()).hasSize(1);
    AnalysisError analysisError = context.allAnalysisErrors().iterator().next();
    assertThat(analysisError.inputFile().filename()).isEqualTo("parse_error.py");
    TextPointer location = analysisError.location();
    assertThat(location).isNotNull();
    assertThat(location.line()).isEqualTo(2);
  }

  @Test
  public void cancelled_analysis() {
    InputFile inputFile = inputFile(FILE_1);
    activeRules = (new ActiveRulesBuilder()).build();
    context.setCancelled(true);
    sensor(null).execute(context);
    assertThat(context.measure(inputFile.key(), CoreMetrics.NCLOC)).isNull();
    assertThat(context.allAnalysisErrors()).isEmpty();
  }

  private PythonSensor sensor() {
    return sensor(CUSTOM_RULES);
  }

  private PythonSensor sensor(@Nullable PythonCustomRuleRepository[] customRuleRepositories) {
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(InputFile.class))).thenReturn(fileLinesContext);
    CheckFactory checkFactory = new CheckFactory(activeRules);
    if(customRuleRepositories == null) {
      return new PythonSensor(fileLinesContextFactory, checkFactory, mock(NoSonarFilter.class));
    }
    return new PythonSensor(fileLinesContextFactory, checkFactory, mock(NoSonarFilter.class), customRuleRepositories);
  }

  private InputFile inputFile(String name) {
    DefaultInputFile inputFile = createInputFile(name);
    context.fileSystem().add(inputFile);
    return inputFile;
  }

  private DefaultInputFile createInputFile(String name) {
    return TestInputFileBuilder.create("moduleKey", name)
        .setModuleBaseDir(baseDir.toPath())
        .setCharset(StandardCharsets.UTF_8)
        .setType(Type.MAIN)
        .setLanguage(Python.KEY)
        .initMetadata(TestUtils.fileContent(new File(baseDir, name), StandardCharsets.UTF_8))
        .build();
  }

  private void verifyUsages(String componentKey, int line, int offset, TextRange... trs) {
    Collection<TextRange> textRanges = context.referencesForSymbolAt(componentKey, line, offset);
    assertThat(textRanges).containsExactly(trs);
  }

  private static TextRange reference(int lineStart, int columnStart, int lineEnd, int columnEnd) {
    return new DefaultTextRange(new DefaultTextPointer(lineStart, columnStart), new DefaultTextPointer(lineEnd, columnEnd));
  }
}
