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
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import javax.annotation.Nullable;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.Mockito;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.event.Level;
import org.sonar.api.SonarProduct;
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
import org.sonar.api.batch.sensor.cpd.internal.TokensLine;
import org.sonar.api.batch.sensor.error.AnalysisError;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.batch.sensor.issue.Issue;
import org.sonar.api.batch.sensor.issue.IssueLocation;
import org.sonar.api.batch.sensor.issue.fix.NewQuickFix;
import org.sonar.api.batch.sensor.issue.fix.QuickFix;
import org.sonar.api.batch.sensor.issue.fix.TextEdit;
import org.sonar.api.config.internal.MapSettings;
import org.sonar.api.internal.SonarRuntimeImpl;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.rule.RuleKey;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.api.utils.Version;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCustomRuleRepository;
import org.sonar.plugins.python.api.PythonInputFileContext;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.internal.EndOfAnalysis;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.caching.Caching;
import org.sonar.plugins.python.caching.TestReadCache;
import org.sonar.plugins.python.caching.TestWriteCache;
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.plugins.python.indexer.SonarLintPythonIndexer;
import org.sonar.plugins.python.indexer.TestModuleFileSystem;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.caching.CpdSerializer;
import org.sonar.python.checks.CheckList;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.tree.TokenImpl;
import org.sonar.python.types.TypeShed;
import org.sonarsource.sonarlint.core.analysis.api.ClientInputFile;
import org.sonarsource.sonarlint.core.analysis.container.analysis.filesystem.FileMetadata;
import org.sonarsource.sonarlint.core.analysis.container.analysis.filesystem.SonarLintInputFile;
import org.sonarsource.sonarlint.core.commons.api.SonarLanguage;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.sonar.plugins.python.caching.Caching.CACHE_VERSION_KEY;
import static org.sonar.plugins.python.caching.Caching.CPD_TOKENS_CACHE_KEY_PREFIX;
import static org.sonar.plugins.python.caching.Caching.CPD_TOKENS_STRING_TABLE_KEY_PREFIX;
import static org.sonar.plugins.python.caching.Caching.IMPORTS_MAP_CACHE_KEY_PREFIX;
import static org.sonar.plugins.python.caching.Caching.PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX;
import static org.sonar.plugins.python.caching.Caching.fileContentHashCacheKey;
import static org.sonar.plugins.python.caching.Caching.importsMapCacheKey;
import static org.sonar.plugins.python.caching.Caching.projectSymbolTableCacheKey;
import static org.sonar.python.index.DescriptorsToProtobuf.toProtobufModuleDescriptor;

class PythonSensorTest {

  private static final String FILE_1 = "file1.py";
  private static final String FILE_2 = "file2.py";
  private static final String FILE_QUICKFIX = "file_quickfix.py";
  private static final String FILE_TEST_FILE = "test_file.py";
  private static final String FILE_INVALID_SYNTAX = "invalid_syntax.py";
  private static final String ONE_STATEMENT_PER_LINE_RULE_KEY = "OneStatementPerLine";
  private static final String FILE_COMPLEXITY_RULE_KEY = "FileComplexity";
  private static final String CUSTOM_REPOSITORY_KEY = "customKey";
  private static final String CUSTOM_RULE_KEY = "key";
  private static final String RULE_CRASHING_ON_SCAN_KEY = "key2";

  private static final Version SONARLINT_DETECTABLE_VERSION = Version.create(6, 0);

  static final SonarRuntime SONARLINT_RUNTIME = SonarRuntimeImpl.forSonarLint(SONARLINT_DETECTABLE_VERSION);

  private static final PythonCustomRuleRepository[] CUSTOM_RULES = {new PythonCustomRuleRepository() {
    @Override
    public String repositoryKey() {
      return CUSTOM_REPOSITORY_KEY;
    }

    @Override
    public List<Class<?>> checkClasses() {
      return List.of(MyCustomRule.class, RuleCrashingOnRegularScan.class);
    }
  }};
  private static Path workDir;

  @Rule(
    key = CUSTOM_RULE_KEY,
    name = "name",
    description = "desc",
    tags = {"bug"})
  public static class MyCustomRule implements PythonCheck, EndOfAnalysis {

    private static final Logger LOG = LoggerFactory.getLogger(MyCustomRule.class);

    @RuleProperty(
      key = "customParam",
      description = "Custom parameter",
      defaultValue = "value")
    public String customParam = "value";

    @Override
    public void scanFile(PythonVisitorContext visitorContext) {
      // do nothing
    }

    @Override
    public boolean scanWithoutParsing(PythonInputFileContext inputFile) {
      return false;
    }

    @Override
    public void endOfAnalysis(CacheContext cacheContext) {
      LOG.trace("End of analysis called!");
    }
  }

  @Rule(
    key = RULE_CRASHING_ON_SCAN_KEY,
    name = "rule_crashing_on_scan",
    description = "desc",
    tags = {"bug"})
  public static class RuleCrashingOnRegularScan implements PythonCheck {

    @Override
    public void scanFile(PythonVisitorContext visitorContext) {
      throw new IllegalStateException("Should not be executed!");
    }

    @Override
    public boolean scanWithoutParsing(PythonInputFileContext inputFile) {
      return true;
    }
  }

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/sensor").getAbsoluteFile();

  private SensorContextTester context;

  private ActiveRules activeRules;

  private final AnalysisWarningsWrapper analysisWarning = mock(AnalysisWarningsWrapper.class);

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);
  @RegisterExtension
  public LogTesterJUnit5 traceLogTester = new LogTesterJUnit5().setLevel(Level.TRACE);

  @BeforeEach
  void init() throws IOException {
    context = SensorContextTester.create(baseDir);
    workDir = Files.createTempDirectory("workDir");
    context.fileSystem().setWorkDir(workDir);
    ProjectPythonVersion.setCurrentVersions(PythonVersionUtils.allVersions());
    TypeShed.resetBuiltinSymbols();
  }

  @Test
  void sensor_descriptor() {
    activeRules = new ActiveRulesBuilder().build();
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    sensor().describe(descriptor);

    assertThat(descriptor.name()).isEqualTo("Python Sensor");
    assertThat(descriptor.languages()).containsOnly("py");
    assertThat(descriptor.type()).isNull();
  }

  @Test
  void test_execute_on_sonarlint() {
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
    assertThat(context.measure(key, CoreMetrics.NCLOC)).isNull();
    assertThat(context.allIssues()).hasSize(1);
    assertThat(context.highlightingTypeAt(key, 15, 2)).isEmpty();
    assertThat(context.allAnalysisErrors()).isEmpty();

    assertThat(PythonScanner.getWorkingDirectory(context)).isNull();
  }

  @Test
  void test_execute_on_sonarlint_quickfix() throws IOException {
    context.setRuntime(SONARLINT_RUNTIME);
    context = Mockito.spy(context);
    when(context.newIssue()).thenReturn(new MockSonarLintIssue(context));

    activate_rule_S2710();
    setup_quickfix_sensor();

    assertThat(context.allIssues()).hasSize(1);

    Collection<Issue> issues = context.allIssues();
    Issue issue = issues.iterator().next();

    var quickFixes = ((MockSonarLintIssue) issue).quickFixes();

    assertThat(quickFixes).hasSize(2);

    QuickFix quickfix = quickFixes.get(0);
    assertThat(quickfix.message()).isEqualTo("Add 'cls' as the first argument.");
    assertThat(quickfix.inputFileEdits()).hasSize(1);
    QuickFix quickfix2 = quickFixes.get(1);
    assertThat(quickfix2.message()).isEqualTo("Rename 'bob' to 'cls'");
    assertThat(quickfix2.inputFileEdits()).hasSize(1);

    List<TextEdit> textEdits = quickfix.inputFileEdits().get(0).textEdits();
    assertThat(textEdits).hasSize(1);
    assertThat(textEdits.get(0).newText()).isEqualTo("cls, ");

    TextRange textRange = reference(4, 13, 4, 13);
    assertThat(textEdits.get(0).range()).usingRecursiveComparison().isEqualTo(textRange);
  }

  @Test
  void test_execute_on_sonarlint_quickfix_broken() throws IOException {
    context.setRuntime(SONARLINT_RUNTIME);
    context = Mockito.spy(context);
    when(context.newIssue()).thenReturn(new MockSonarLintIssue(context) {
      @Override
      public NewQuickFix newQuickFix() {
        throw new RuntimeException("Exception message");
      }
    });

    activate_rule_S2710();
    setup_quickfix_sensor();

    Collection<Issue> issues = context.allIssues();
    assertThat(issues).hasSize(1);
    MockSonarLintIssue issue = (MockSonarLintIssue) issues.iterator().next();

    assertThat(issue.quickFixes()).isEmpty();
    assertThat(issue.getSaved()).isTrue();

    assertThat(traceLogTester.logs()).contains("Could not report quick fixes for rule: python:S2710. java.lang.RuntimeException: Exception message");
  }

  @Test
  void test_symbol_visitor() {
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
  void test_issues() {
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

    PythonInputFile inputFile = inputFile(FILE_2);
    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(3);
    Iterator<Issue> issuesIterator = context.allIssues().iterator();

    int checkedIssues = 0;

    while (issuesIterator.hasNext()) {
      Issue issue = issuesIterator.next();
      IssueLocation issueLocation = issue.primaryLocation();
      assertThat(issueLocation.inputComponent()).isEqualTo(inputFile.wrappedFile());

      switch (issue.ruleKey().rule()) {
        case "S134":
          assertThat(issueLocation.message()).isEqualTo("Refactor this code to not nest more than 4 \"if\", \"for\", \"while\", \"try\" and \"with\" statements.");
          assertThat(issueLocation.textRange()).isEqualTo(inputFile.wrappedFile().newRange(7, 16, 7, 18));
          assertThat(issue.flows()).hasSize(4);
          assertThat(issue.gap()).isNull();
          checkedIssues++;
          break;
        case ONE_STATEMENT_PER_LINE_RULE_KEY:
          assertThat(issueLocation.message()).isEqualTo("At most one statement is allowed per line, but 2 statements were found on this line.");
          assertThat(issueLocation.textRange()).isEqualTo(inputFile.wrappedFile().newRange(1, 0, 1, 50));
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
    assertThat(logTester.logs(Level.INFO)).contains("Starting global symbols computation");
    assertThat(logTester.logs(Level.INFO)).contains("Starting rules execution");
    assertThat(logTester.logs(Level.INFO).stream().filter(line -> line.equals("1 source file to be analyzed")).count()).isEqualTo(2);

    assertThat(PythonScanner.getWorkingDirectory(context)).isEqualTo(workDir.toFile());
  }

  @Test
  void cross_files_secondary_locations() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S930"))
        .build())
      .build();

    PythonInputFile mainFile = inputFile("main.py");
    PythonInputFile modFile = inputFile("mod.py");
    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(1);
    Issue issue = context.allIssues().iterator().next();
    assertThat(issue.flows()).hasSize(1);
    Issue.Flow flow = issue.flows().get(0);
    assertThat(flow.locations()).hasSize(2);
    assertThat(flow.locations().get(0).inputComponent()).isEqualTo(mainFile.wrappedFile());
    assertThat(flow.locations().get(1).inputComponent()).isEqualTo(modFile.wrappedFile());
  }

  @Test
  void no_cross_file_issues_only_one_file() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S930"))
        .build())
      .build();

    inputFile("main.py");
    sensor().execute(context);
    assertThat(context.allIssues()).isEmpty();
  }

  @Test
  void cross_files_issues_only_one_file_analyzed() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S930"))
        .build())
      .build();

    PythonInputFile mainFile = inputFile("main.py");
    // "mod.py" created but not added to context
    PythonInputFile modFile = createInputFile("mod.py");
    PythonIndexer pythonIndexer = pythonIndexer(Arrays.asList(mainFile, modFile));
    sensor(null, pythonIndexer, analysisWarning).execute(context);
    assertThat(context.allIssues()).hasSize(1);
    Issue issue = context.allIssues().iterator().next();
    assertThat(issue.primaryLocation().inputComponent()).isEqualTo(mainFile.wrappedFile());
    assertThat(issue.flows()).hasSize(1);
    Issue.Flow flow = issue.flows().get(0);
    assertThat(flow.locations()).hasSize(2);
    assertThat(flow.locations().get(0).inputComponent()).isEqualTo(mainFile.wrappedFile());
    assertThat(flow.locations().get(1).inputComponent()).isEqualTo(modFile.wrappedFile());
  }

  @Test
  void not_relying_on_stubs_for_project_under_analysis() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S6709"))
        .build())
      .build();

    PythonInputFile firstFile = inputFile("sklearn/__init__.py");
    PythonInputFile secondFile = inputFile("sklearn/my_file.py");
    PythonIndexer pythonIndexer = pythonIndexer(Arrays.asList(firstFile, secondFile));
    sensor(null, pythonIndexer, analysisWarning).execute(context);
    assertThat(context.allIssues()).isEmpty();
  }

  @Test
  void not_relying_on_stubs_for_project_under_analysis_2() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S6709"))
        .build())
      .build();

    PythonInputFile initFile = inputFile("not_sklearn/__init__.py");
    PythonInputFile mainFile = inputFile("not_sklearn/my_file.py");
    PythonIndexer pythonIndexer = pythonIndexer(Arrays.asList(initFile, mainFile));
    sensor(null, pythonIndexer, analysisWarning).execute(context);
    assertThat(context.allIssues()).hasSize(1);
    Issue issue = context.allIssues().iterator().next();
    assertThat(issue.primaryLocation().inputComponent()).isEqualTo(mainFile.wrappedFile());
    assertThat(issue.ruleKey().rule()).isEqualTo("S6709");
  }

  @Test
  void end_of_analysis_called() {
    inputFile(FILE_2);
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CUSTOM_REPOSITORY_KEY, CUSTOM_RULE_KEY))
        .build())
      .build();
    sensor().execute(context);

    assertThat(traceLogTester.logs(Level.TRACE)).containsExactly("End of analysis called!");
  }

  @Test
  void no_indexer_when_project_too_large_sonarlint() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S930"))
        .build())
      .build();
    context.setSettings(new MapSettings().setProperty("sonar.python.sonarlint.indexing.maxlines", 1));

    PythonInputFile mainFile = inputFile("main.py");
    PythonIndexer pythonIndexer = pythonIndexer(Collections.singletonList(mainFile));
    sensor(CUSTOM_RULES, pythonIndexer, analysisWarning).execute(context);
    assertThat(context.allIssues()).isEmpty();
    assertThat(logTester.logs(Level.DEBUG)).contains("Project symbol table deactivated due to project size (total number of lines is 4, maximum for indexing is 1)");
    assertThat(logTester.logs(Level.DEBUG)).contains("Update \"sonar.python.sonarlint.indexing.maxlines\" to set a different limit.");
  }

  @Test
  void loop_in_class_hierarchy() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S2710"))
        .build())
      .build();

    PythonInputFile mainFile = inputFile("modA.py");
    PythonInputFile modFile = inputFile("modB.py");
    PythonIndexer pythonIndexer = pythonIndexer(Arrays.asList(mainFile, modFile));
    sensor(null, pythonIndexer, analysisWarning).execute(context);

    assertThat(context.allIssues()).hasSize(1);
  }

  @Test
  void test_issues_on_test_files() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S1226"))
        .build())
      .build();

    PythonInputFile inputFile = inputFile(FILE_TEST_FILE, Type.TEST);
    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(1);
    Issue issue = context.allIssues().iterator().next();
    assertThat(issue.primaryLocation().inputComponent()).isEqualTo(inputFile.wrappedFile());
    assertThat(issue.ruleKey().rule()).isEqualTo("S5905");
  }

  @Test
  void test_failFast_triggered_on_main_files() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .build();

    inputFile(FILE_INVALID_SYNTAX, Type.MAIN);
    context.setSettings(new MapSettings().setProperty("sonar.internal.analysis.failFast", true));
    PythonSensor sensor = sensor();
    assertThatThrownBy(() -> sensor.execute(context)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void test_failFast_not_triggered_on_test_files() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .build();

    inputFile(FILE_INVALID_SYNTAX, Type.TEST);
    context.setSettings(new MapSettings().setProperty("sonar.internal.analysis.failFast", true));
    sensor().execute(context);
    assertThat(context.allIssues()).isEmpty();
  }

  @Test
  void test_test_file_highlighting() {
    activeRules = new ActiveRulesBuilder().build();

    DefaultInputFile inputFile1 = spy(TestInputFileBuilder.create("moduleKey", FILE_1)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(UTF_8)
      .setType(Type.TEST)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, FILE_1), UTF_8))
      .build());

    DefaultInputFile inputFile2 = spy(TestInputFileBuilder.create("moduleKey", FILE_2)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(UTF_8)
      .setType(Type.TEST)
      .setLanguage(Python.KEY)
      .build());

    DefaultInputFile inputFile3 = spy(TestInputFileBuilder.create("moduleKey", "parse_error.py")
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(UTF_8)
      .setType(Type.TEST)
      .setLanguage(Python.KEY)
      .build());

    context.fileSystem().add(inputFile1);
    context.fileSystem().add(inputFile2);
    context.fileSystem().add(inputFile3);
    sensor().execute(context);
    assertThat(logTester.logs()).contains("Unable to parse file: parse_error.py");
    assertThat(logTester.logs()).contains("Unable to analyze file: file2.py");
    assertThat(context.highlightingTypeAt(inputFile1.key(), 1, 2)).isNotEmpty();
  }

  @Test
  void test_exception_does_not_fail_analysis() throws IOException {
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
      .setCharset(UTF_8)
      .setType(Type.MAIN)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, FILE_1), UTF_8))
      .setStatus(InputFile.Status.ADDED)
      .build());
    when(inputFile.contents()).thenThrow(RuntimeException.class);

    context.fileSystem().add(inputFile);
    inputFile(FILE_2);

    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(2);
  }

  @Test
  void test_exception_should_fail_analysis_if_configured_so() throws IOException {
    DefaultInputFile inputFile = (DefaultInputFile) spy(createInputFile(FILE_1).wrappedFile());
    when(inputFile.contents()).thenThrow(FileNotFoundException.class);
    context.fileSystem().add(inputFile);

    activeRules = new ActiveRulesBuilder().build();
    context.setSettings(new MapSettings().setProperty("sonar.internal.analysis.failFast", "true"));
    PythonSensor sensor = sensor();

    assertThatThrownBy(() -> sensor.execute(context))
      .isInstanceOf(IllegalStateException.class)
      .hasCauseInstanceOf(FileNotFoundException.class);
  }

  @Test
  void test_python_version_parameter_warning() {
    context.fileSystem().add(inputFile(FILE_1).wrappedFile());

    activeRules = new ActiveRulesBuilder().build();

    sensor().execute(context);
    assertThat(logTester.logs(Level.WARN)).contains(PythonSensor.UNSET_VERSION_WARNING);
    verify(analysisWarning, times(1)).addUnique(PythonSensor.UNSET_VERSION_WARNING);
  }

  @Test
  void test_python_version_parameter_no_warning() {
    context.fileSystem().add(inputFile(FILE_1).wrappedFile());

    activeRules = new ActiveRulesBuilder().build();

    context.setSettings(new MapSettings().setProperty("sonar.python.version", "3.13"));
    sensor().execute(context);
    assertThat(ProjectPythonVersion.currentVersions()).containsExactly(PythonVersionUtils.Version.V_313);
    assertThat(logTester.logs(Level.WARN)).doesNotContain(PythonSensor.UNSET_VERSION_WARNING);
    verify(analysisWarning, times(0)).addUnique(PythonSensor.UNSET_VERSION_WARNING);
  }

  void setup_typing_concise_rule(String pythonVersion) {
    context.fileSystem().add(inputFile("python-version/typing.py").wrappedFile());

    activeRules = new ActiveRulesBuilder().addRule(new NewActiveRule.Builder()
      .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S6546"))
      .build()).build();

    context.setSettings(new MapSettings().setProperty("sonar.python.version", pythonVersion));
    sensor().execute(context);
  }

  @ParameterizedTest
  @ValueSource(strings = {"2.7", "3", "3.1", "3.9"})
  void test_python_version_pre_310(String version) {
    setup_typing_concise_rule(version);

    assertThat(context.allIssues()).isEmpty();
  }

  @ParameterizedTest
  @ValueSource(strings = {"3.10", "3.11", "3.12"})
  void test_python_version_post_310(String version) {
    setup_typing_concise_rule(version);

    assertThat(context.allIssues()).hasSize(1);
  }

  @Test
  void test_python_version_unknown_upper() {
    setup_typing_concise_rule("3.4569");

    assertThat(ProjectPythonVersion.currentVersions()).containsExactly(PythonVersionUtils.MAX_SUPPORTED_VERSION);
    assertThat(context.allIssues()).hasSize(1);
  }

  @Test
  void test_python_version_unknown_lower() {
    setup_typing_concise_rule("2.4569");

    assertThat(ProjectPythonVersion.currentVersions()).containsExactlyElementsOf(PythonVersionUtils.allVersions());
    assertThat(context.allIssues()).isEmpty();
  }

  @Test
  void parse_error() {
    inputFile("parse_error.py");
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "ParsingError"))
        .build())
      .build();

    sensor().execute(context);
    assertThat(context.allIssues()).hasSize(1);
    String log = String.join("\n", logTester.logs());
    assertThat(log).contains("Parse error at line 2")
      .doesNotContain("java.lang.NullPointerException");
    assertThat(context.allAnalysisErrors()).hasSize(1);
    AnalysisError analysisError = context.allAnalysisErrors().iterator().next();
    assertThat(analysisError.inputFile().filename()).isEqualTo("parse_error.py");
    TextPointer location = analysisError.location();
    assertThat(location).isNotNull();
    assertThat(location.line()).isEqualTo(2);
  }

  @Test
  void cancelled_analysis() {
    PythonInputFile inputFile = inputFile(FILE_1);
    activeRules = (new ActiveRulesBuilder()).build();
    context.setCancelled(true);
    sensor(null, null, analysisWarning).execute(context);
    assertThat(context.measure(inputFile.wrappedFile().key(), CoreMetrics.NCLOC)).isNull();
    assertThat(context.allAnalysisErrors()).isEmpty();
  }

  @Test
  void saving_performance_measure_not_activated_by_default() {
    activeRules = (new ActiveRulesBuilder()).build();

    inputFile("main.py");
    sensor().execute(context);
    assertThat(context.allIssues()).isEmpty();
    assertThat(logTester.logs(Level.INFO)).noneMatch(s -> s.matches(".*performance measures.*"));
    Path defaultPerformanceFile = workDir.resolve("sonar-python-performance-measure.json");
    assertThat(defaultPerformanceFile).doesNotExist();
  }

  @Test
  void saving_performance_measure() throws IOException {
    context.setSettings(new MapSettings().setProperty("sonar.python.performance.measure", "true"));
    activeRules = (new ActiveRulesBuilder()).build();

    inputFile("main.py");
    sensor().execute(context);
    Path defaultPerformanceFile = workDir.resolve("sonar-python-performance-measure.json");
    assertThat(logTester.logs(Level.INFO)).anyMatch(s -> s.matches(".*performance measures.*"));
    assertThat(defaultPerformanceFile).exists();
    assertThat(new String(Files.readAllBytes(defaultPerformanceFile), UTF_8)).contains("\"PythonSensor\"");
  }

  @Test
  void saving_performance_measure_custom_path() throws IOException {
    Path customPerformanceFile = workDir.resolve("custom.performance.measure.json");
    MapSettings mapSettings = new MapSettings();
    mapSettings.setProperty("sonar.python.performance.measure", "true");
    mapSettings.setProperty("sonar.python.performance.measure.path", customPerformanceFile.toString());
    context.setSettings(mapSettings);
    activeRules = (new ActiveRulesBuilder()).build();

    inputFile("main.py");
    sensor().execute(context);
    assertThat(logTester.logs(Level.INFO)).anyMatch(s -> s.matches(".*performance measures.*"));
    Path defaultPerformanceFile = workDir.resolve("sonar-python-performance-measure.json");
    assertThat(defaultPerformanceFile).doesNotExist();
    assertThat(customPerformanceFile).exists();
    assertThat(new String(Files.readAllBytes(customPerformanceFile), UTF_8)).contains("\"PythonSensor\"");
  }

  @Test
  void saving_performance_measure_empty_path() throws IOException {
    MapSettings mapSettings = new MapSettings();
    mapSettings.setProperty("sonar.python.performance.measure", "true");
    mapSettings.setProperty("sonar.python.performance.measure.path", "");
    context.setSettings(mapSettings);
    activeRules = (new ActiveRulesBuilder()).build();

    inputFile("main.py");
    sensor().execute(context);
    assertThat(logTester.logs(Level.INFO)).anyMatch(s -> s.matches(".*performance measures.*"));
    Path defaultPerformanceFile = workDir.resolve("sonar-python-performance-measure.json");
    assertThat(defaultPerformanceFile).exists();
    assertThat(new String(Files.readAllBytes(defaultPerformanceFile), UTF_8)).contains("\"PythonSensor\"");
  }

  @Test
  void test_using_cache() throws IOException {
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

    PythonInputFile inputFile = inputFile(FILE_2, Type.MAIN, InputFile.Status.SAME);
    TestReadCache readCache = getValidReadCache();
    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    CpdSerializer.SerializationResult cpdTokens = CpdSerializer.serialize(Collections.emptyList());
    readCache.put(importsMapCacheKey(inputFile.wrappedFile().key()), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey(inputFile.wrappedFile().key()), serializedSymbolTable);
    readCache.put(CPD_TOKENS_CACHE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.data);
    readCache.put(CPD_TOKENS_STRING_TABLE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.stringTable);
    readCache.put(fileContentHashCacheKey(inputFile.wrappedFile().key()), inputFile.wrappedFile().md5Hash().getBytes(UTF_8));
    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));
    sensor().execute(context);

    assertThat(context.allIssues()).isEmpty();
    assertThat(logTester.logs(Level.INFO))
      .contains("The Python analyzer was able to leverage cached data from previous analyses for 1 out of 1 files. These files were not parsed.");
  }

  @Test
  void test_scan_without_parsing_test_file() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S1226"))
        .build())
      .build();

    PythonInputFile inputFile = inputFile(FILE_TEST_FILE, Type.TEST, InputFile.Status.SAME);
    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("test_func", "test_file.test_func", null))).toByteArray();
    TestReadCache readCache = getValidReadCache();
    readCache.put(fileContentHashCacheKey(inputFile.wrappedFile().key()), inputFile.wrappedFile().md5Hash().getBytes(UTF_8));
    readCache.put(importsMapCacheKey(inputFile.wrappedFile().key()), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey(inputFile.wrappedFile().key()), serializedSymbolTable);
    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));
    sensor().execute(context);

    assertThat(context.allIssues()).isEmpty();
  }

  @Test
  void test_scan_without_parsing_fails_does_not_reexecute_successful_checks() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, ONE_STATEMENT_PER_LINE_RULE_KEY))
        .build())
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CUSTOM_REPOSITORY_KEY, CUSTOM_RULE_KEY))
        .build())
      .build();

    PythonInputFile inputFile = inputFile(FILE_2, Type.MAIN, InputFile.Status.SAME);
    TestReadCache readCache = getValidReadCache();
    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    readCache.put(importsMapCacheKey(inputFile.wrappedFile().key()), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey(inputFile.wrappedFile().key()), serializedSymbolTable);
    readCache.put(fileContentHashCacheKey(inputFile.wrappedFile().key()), inputFile.wrappedFile().md5Hash().getBytes(UTF_8));
    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));
    sensor().execute(context);

    assertThat(context.allIssues()).isEmpty();
    assertThat(logTester.logs(Level.INFO))
      .contains("The Python analyzer was able to leverage cached data from previous analyses for 0 out of 1 files. These files were not parsed.");
  }

  @Test
  void test_partial_scan_without_parsing() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, ONE_STATEMENT_PER_LINE_RULE_KEY))
        .build())
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CUSTOM_REPOSITORY_KEY, CUSTOM_RULE_KEY))
        .build())
      .build();

    inputFile(FILE_1, Type.MAIN, InputFile.Status.CHANGED);
    PythonInputFile inputFile2 = inputFile(FILE_2, Type.MAIN, InputFile.Status.SAME);
    TestReadCache readCache = getValidReadCache();
    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    readCache.put(importsMapCacheKey(inputFile2.wrappedFile().key()), String.join(";", List.of("file1.py")).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey(inputFile2.wrappedFile().key()), serializedSymbolTable);
    readCache.put(fileContentHashCacheKey(inputFile2.wrappedFile().key()), inputFile2.wrappedFile().md5Hash().getBytes(UTF_8));
    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));
    sensor().execute(context);

    assertThat(context.allIssues()).hasSize(2);
    assertThat(logTester.logs(Level.INFO))
      .contains("The Python analyzer was able to leverage cached data from previous analyses for 0 out of 2 files. These files were not parsed.");
  }

  @Test
  void cache_not_enabled_for_older_api_version() {
    SensorContextTester contextMock = spy(context);
    SonarRuntime runtime = mock(SonarRuntime.class);
    when(contextMock.runtime()).thenReturn(runtime);
    when(runtime.getProduct()).thenReturn(SonarProduct.SONARQUBE);
    when(runtime.getApiVersion()).thenReturn(Version.create(9, 6));
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, ONE_STATEMENT_PER_LINE_RULE_KEY))
        .build())
      .build();

    inputFile(FILE_2, Type.MAIN, InputFile.Status.SAME);
    TestReadCache readCache = new TestReadCache();
    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Set.of(new VariableDescriptor("x", "main.x", null))).toByteArray();
    readCache.put(IMPORTS_MAP_CACHE_KEY_PREFIX + "file2", String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(PROJECT_SYMBOL_TABLE_CACHE_KEY_PREFIX + "file2", serializedSymbolTable);
    sensor().execute(contextMock);

    assertThat(context.allIssues()).hasSize(1);
  }

  @Test
  void write_cpd_tokens_to_cache() throws IOException {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .build();

    TestReadCache readCache = getValidReadCache();
    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    inputFile("pass.py", Type.MAIN, InputFile.Status.ADDED);
    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));
    sensor().execute(context);

    assertThat(writeCache.getData().keySet()).containsExactlyInAnyOrder(
      "python:cache_version", "python:files", "python:descriptors:moduleKey:pass.py", "python:imports:moduleKey:pass.py",
      "python:cpd:data:moduleKey:pass.py", "python:cpd:stringTable:moduleKey:pass.py", "python:content_hashes:moduleKey:pass.py");

    byte[] tokenData = writeCache.getData().get("python:cpd:data:moduleKey:pass.py");
    byte[] stringTable = writeCache.getData().get("python:cpd:stringTable:moduleKey:pass.py");

    List<CpdSerializer.TokenInfo> actualTokens = CpdSerializer.deserialize(tokenData, stringTable);
    assertThat(actualTokens)
      .hasSize(1);

    assertThat(actualTokens.get(0))
      .usingRecursiveComparison()
      .isEqualTo(new CpdSerializer.TokenInfo(1, 0, 1, 4, "pass"));
  }

  @Test
  void write_cpd_tokens_to_cache_failure() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .build();

    TestReadCache readCache = getValidReadCache();
    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    PythonInputFile inputFile = inputFile("pass.py", Type.MAIN, InputFile.Status.ADDED);
    writeCache.write(CPD_TOKENS_CACHE_KEY_PREFIX + inputFile.wrappedFile().key(), "whatever".getBytes());

    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));
    sensor().execute(context);

    assertThat(logTester.logs(Level.WARN))
      .contains("Could not write CPD tokens to cache (IllegalArgumentException: Same key cannot be written to multiple times (python:cpd:data:moduleKey:pass.py))");
  }

  @Test
  void write_cpd_tokens_multiple_files() throws IOException {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .build();

    TestReadCache readCache = getValidReadCache();
    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    inputFile("main.py", Type.MAIN, InputFile.Status.ADDED);
    inputFile("pass.py", Type.MAIN, InputFile.Status.ADDED);
    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));
    sensor().execute(context);

    byte[] mainTokensData = writeCache.getData().get("python:cpd:data:moduleKey:main.py");
    byte[] mainTokensTable = writeCache.getData().get("python:cpd:stringTable:moduleKey:main.py");
    List<CpdSerializer.TokenInfo> actualTokensForMain = CpdSerializer.deserialize(mainTokensData, mainTokensTable);
    assertThat(actualTokensForMain)
      .hasSize(14);

    byte[] passTokensData = writeCache.getData().get("python:cpd:data:moduleKey:pass.py");
    byte[] passTokensTable = writeCache.getData().get("python:cpd:stringTable:moduleKey:pass.py");
    List<CpdSerializer.TokenInfo> actualTokensForPass = CpdSerializer.deserialize(passTokensData, passTokensTable);
    assertThat(actualTokensForPass)
      .hasSize(1);
  }

  @Test
  void read_cpd_tokens_from_cache() throws IOException {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .build();

    PythonInputFile inputFile = inputFile("main.py", Type.MAIN, InputFile.Status.SAME);
    var sslrToken = passToken(inputFile.wrappedFile().uri());
    List<Token> tokens = List.of(new TokenImpl(sslrToken));

    TestReadCache readCache = getValidReadCache();
    CpdSerializer.SerializationResult cpdTokens = CpdSerializer.serialize(tokens);
    readCache.put(CPD_TOKENS_CACHE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.data);
    readCache.put(CPD_TOKENS_STRING_TABLE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.stringTable);
    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Collections.emptySet()).toByteArray();
    readCache.put(importsMapCacheKey(inputFile.wrappedFile().key()), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey(inputFile.wrappedFile().key()), serializedSymbolTable);
    readCache.put(fileContentHashCacheKey(inputFile.wrappedFile().key()), inputFile.wrappedFile().md5Hash().getBytes(UTF_8));

    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));

    sensor().execute(context);

    // Verify the written CPD tokens
    List<TokensLine> tokensLines = context.cpdTokens("moduleKey:main.py");
    assertThat(tokensLines)
      .isNotNull()
      .hasSize(1);

    assertThat(tokensLines.get(0).getValue()).isEqualTo("pass");

    // Verify that we carried the tokens over to the next cache
    assertThat(writeCache.getData())
      .containsEntry(CPD_TOKENS_CACHE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.data)
      .containsEntry(CPD_TOKENS_STRING_TABLE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.stringTable);
  }

  @Test
  void read_cpd_tokens_from_cache_not_in_cache() throws IOException {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .build();

    PythonInputFile inputFile = inputFile("pass.py", Type.MAIN, InputFile.Status.SAME);

    TestReadCache readCache = getValidReadCache();
    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Collections.emptySet()).toByteArray();
    readCache.put(importsMapCacheKey(inputFile.wrappedFile().key()), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey(inputFile.wrappedFile().key()), serializedSymbolTable);

    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));

    sensor().execute(context);

    // Verify the written CPD tokens
    List<TokensLine> tokensLines = context.cpdTokens("moduleKey:pass.py");
    assertThat(tokensLines)
      .isNotNull()
      .hasSize(1);

    assertThat(tokensLines.get(0).getValue()).isEqualTo("pass");

    // Verify that we carried the tokens over to the next cache
    List<Token> expectedTokens = List.of(new TokenImpl(passToken(inputFile.wrappedFile().uri())));
    CpdSerializer.SerializationResult cpdTokens = CpdSerializer.serialize(expectedTokens);

    assertThat(writeCache.getData())
      .containsEntry(Caching.CPD_TOKENS_CACHE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.data)
      .containsEntry(CPD_TOKENS_STRING_TABLE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.stringTable);
  }

  @Test
  void read_cpd_tokens_from_cache_corrupted_format() throws IOException {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .build();

    PythonInputFile inputFile = inputFile("pass.py", Type.MAIN, InputFile.Status.SAME);
    var sslrToken = passToken(inputFile.wrappedFile().uri());
    List<Token> tokens = List.of(new TokenImpl(sslrToken));

    TestReadCache readCache = getValidReadCache();

    readCache.put(CPD_TOKENS_CACHE_KEY_PREFIX + inputFile.wrappedFile().key(), "not valid data".getBytes(UTF_8));
    readCache.put(CPD_TOKENS_STRING_TABLE_KEY_PREFIX + inputFile.wrappedFile().key(), "not valid string table".getBytes(UTF_8));

    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Collections.emptySet()).toByteArray();
    readCache.put(importsMapCacheKey(inputFile.wrappedFile().key()), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey(inputFile.wrappedFile().key()), serializedSymbolTable);
    readCache.put(fileContentHashCacheKey(inputFile.wrappedFile().key()), inputFile.wrappedFile().md5Hash().getBytes(UTF_8));

    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));

    sensor().execute(context);

    assertThat(logTester.logs(Level.WARN))
      .anyMatch(line -> line.startsWith("Failed to deserialize CPD tokens"));

    // Verify the written CPD tokens
    List<TokensLine> tokensLines = context.cpdTokens("moduleKey:pass.py");
    assertThat(tokensLines)
      .isNotNull()
      .hasSize(1);

    assertThat(tokensLines.get(0).getValue()).isEqualTo("pass");

    // Verify that we carried the tokens over to the next cache
    List<Token> expectedTokens = List.of(new TokenImpl(passToken(inputFile.wrappedFile().uri())));
    CpdSerializer.SerializationResult cpdTokens = CpdSerializer.serialize(expectedTokens);

    assertThat(writeCache.getData())
      .containsEntry(Caching.CPD_TOKENS_CACHE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.data)
      .containsEntry(CPD_TOKENS_STRING_TABLE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.stringTable);
  }

  @Test
  void read_cpd_tokens_cache_disabled() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S5905"))
        .build())
      .build();

    TestReadCache readCache = getValidReadCache();
    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    inputFile("pass.py", Type.MAIN, InputFile.Status.SAME);
    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(false);
    context.setSettings(new MapSettings().setProperty("sonar.python.skipUnchanged", true));
    sensor().execute(context);

    List<TokensLine> tokensLines = context.cpdTokens("moduleKey:pass.py");
    assertThat(tokensLines)
      .isNotNull()
      .hasSize(1);

    assertThat(tokensLines.get(0).getValue()).isEqualTo("pass");
  }

  @Test
  void cpd_tokens_failure_does_not_execute_checks_multiple_times() throws IOException {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CUSTOM_REPOSITORY_KEY, RULE_CRASHING_ON_SCAN_KEY))
        .build())
      .build();

    PythonInputFile inputFile = inputFile("pass.py", Type.MAIN, InputFile.Status.SAME);

    TestReadCache readCache = getValidReadCache();
    byte[] serializedSymbolTable = toProtobufModuleDescriptor(Collections.emptySet()).toByteArray();
    readCache.put(importsMapCacheKey(inputFile.wrappedFile().key()), String.join(";", Collections.emptyList()).getBytes(StandardCharsets.UTF_8));
    readCache.put(projectSymbolTableCacheKey(inputFile.wrappedFile().key()), serializedSymbolTable);
    readCache.put(fileContentHashCacheKey(inputFile.wrappedFile().key()), inputFile.wrappedFile().md5Hash().getBytes(UTF_8));

    TestWriteCache writeCache = new TestWriteCache();
    writeCache.bind(readCache);

    context.setPreviousCache(readCache);
    context.setNextCache(writeCache);
    context.setCacheEnabled(true);
    context.setSettings(
      new MapSettings()
        .setProperty("sonar.python.skipUnchanged", true)
        .setProperty("sonar.internal.analysis.failFast", true));

    sensor().execute(context);

    // Verify the written CPD tokens
    List<TokensLine> tokensLines = context.cpdTokens("moduleKey:pass.py");
    assertThat(tokensLines)
      .isNotNull()
      .hasSize(1);

    assertThat(tokensLines.get(0).getValue()).isEqualTo("pass");

    // Verify that we carried the tokens over to the next cache
    List<Token> expectedTokens = List.of(new TokenImpl(passToken(inputFile.wrappedFile().uri())));
    CpdSerializer.SerializationResult cpdTokens = CpdSerializer.serialize(expectedTokens);

    assertThat(writeCache.getData())
      .containsEntry(Caching.CPD_TOKENS_CACHE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.data)
      .containsEntry(CPD_TOKENS_STRING_TABLE_KEY_PREFIX + inputFile.wrappedFile().key(), cpdTokens.stringTable);
  }

  @Test
  void test_scanner_isNotebook() {
    var regularPythonFile = mock(PythonInputFile.class);
    when(regularPythonFile.kind()).thenReturn(PythonInputFile.Kind.PYTHON);
    assertThat(PythonScanner.isNotebook(regularPythonFile)).isFalse();

    var notebookPythonFile = mock(PythonInputFile.class);
    when(notebookPythonFile.kind()).thenReturn(PythonInputFile.Kind.IPYTHON);
    assertThat(PythonScanner.isNotebook(notebookPythonFile)).isTrue();
  }

  private com.sonar.sslr.api.Token passToken(URI uri) {
    return com.sonar.sslr.api.Token.builder()
      .setType(PythonKeyword.PASS)
      .setLine(1)
      .setColumn(0)
      .setURI(uri)
      .setValueAndOriginalValue("pass")
      .build();
  }

  private PythonSensor sensor() {
    return sensor(CUSTOM_RULES, null, analysisWarning);
  }

  private PythonSensor sensor(@Nullable PythonCustomRuleRepository[] customRuleRepositories, @Nullable PythonIndexer indexer, AnalysisWarningsWrapper analysisWarnings) {
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(InputFile.class))).thenReturn(fileLinesContext);
    CheckFactory checkFactory = new CheckFactory(activeRules);
    if (indexer == null && customRuleRepositories == null) {
      return new PythonSensor(fileLinesContextFactory, checkFactory, mock(NoSonarFilter.class), analysisWarnings);
    }
    if (indexer == null) {
      return new PythonSensor(fileLinesContextFactory, checkFactory, mock(NoSonarFilter.class), customRuleRepositories, analysisWarnings);
    }
    if (customRuleRepositories == null) {
      return new PythonSensor(fileLinesContextFactory, checkFactory, mock(NoSonarFilter.class), indexer, new SonarLintCache(), analysisWarnings);
    }
    return new PythonSensor(fileLinesContextFactory, checkFactory, mock(NoSonarFilter.class), customRuleRepositories, indexer, new SonarLintCache(), analysisWarnings);
  }

  private SonarLintPythonIndexer pythonIndexer(List<PythonInputFile> files) {
    return new SonarLintPythonIndexer(new TestModuleFileSystem(files));
  }

  private PythonInputFile inputFile(String name) {
    return inputFile(name, Type.MAIN);
  }

  private PythonInputFile inputFile(String name, Type fileType) {
    PythonInputFile inputFile = createInputFile(name, fileType, InputFile.Status.ADDED);
    context.fileSystem().add(inputFile.wrappedFile());
    return inputFile;
  }

  private PythonInputFile inputFile(String name, Type fileType, InputFile.Status status) {
    PythonInputFile inputFile = createInputFile(name, fileType, status);
    context.fileSystem().add(inputFile.wrappedFile());
    return inputFile;
  }

  private PythonInputFile createInputFile(String name) {
    return createInputFile(name, Type.MAIN, InputFile.Status.ADDED);
  }

  private PythonInputFile createInputFile(String name, Type fileType, InputFile.Status status) {
    return new PythonInputFileImpl(TestInputFileBuilder.create("moduleKey", name)
      .setModuleBaseDir(baseDir.toPath())
      .setCharset(UTF_8)
      .setType(fileType)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(new File(baseDir, name), UTF_8))
      .setStatus(status)
      .build());
  }

  private void verifyUsages(String componentKey, int line, int offset, TextRange... trs) {
    Collection<TextRange> textRanges = context.referencesForSymbolAt(componentKey, line, offset);
    assertThat(textRanges).containsExactly(trs);
  }

  private static TextRange reference(int lineStart, int columnStart, int lineEnd, int columnEnd) {
    return new DefaultTextRange(new DefaultTextPointer(lineStart, columnStart), new DefaultTextPointer(lineEnd, columnEnd));
  }

  private void activate_rule_S2710() {
    activeRules = new ActiveRulesBuilder()
      .addRule(new NewActiveRule.Builder()
        .setRuleKey(RuleKey.of(CheckList.REPOSITORY_KEY, "S2710"))
        .setName("First argument to class methods should follow naming convention")
        .build())
      .build();
  }

  private void setup_quickfix_sensor() throws IOException {
    String pathToQuickFixTestFile = "src/test/resources/org/sonar/plugins/python/sensor/" + FILE_QUICKFIX;
    File file = new File(pathToQuickFixTestFile);
    String content = Files.readString(file.toPath());

    ClientInputFile clientFile = mock(ClientInputFile.class);

    when(clientFile.relativePath()).thenReturn(pathToQuickFixTestFile);
    when(clientFile.getPath()).thenReturn(file.getAbsolutePath());
    when(clientFile.uri()).thenReturn(file.getAbsoluteFile().toURI());
    when(clientFile.contents()).thenReturn(content);

    Function<SonarLintInputFile, FileMetadata.Metadata> metadataGenerator = x -> {
      try {
        return new FileMetadata().readMetadata(new FileInputStream(file), StandardCharsets.UTF_8, file.toURI(), null);
      } catch (FileNotFoundException e) {
        throw new RuntimeException(e);
      }
    };

    SonarLintInputFile sonarFile = new SonarLintInputFile(clientFile, metadataGenerator);
    sonarFile.setType(Type.MAIN);
    sonarFile.setLanguage(SonarLanguage.PYTHON);

    context.fileSystem().add(sonarFile);
    sensor().execute(context);
  }

  TestReadCache getValidReadCache() {
    TestReadCache testReadCache = new TestReadCache();
    testReadCache.put(CACHE_VERSION_KEY, "unknownPluginVersion".getBytes(UTF_8));
    return testReadCache;
  }
}
