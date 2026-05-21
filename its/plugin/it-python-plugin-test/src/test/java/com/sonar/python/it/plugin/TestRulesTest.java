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
package com.sonar.python.it.plugin;

import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.python.it.ConcurrentOrchestratorExtension;
import com.sonar.python.it.TestsUtils;
import java.io.File;
import java.util.Comparator;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.sonarqube.ws.Issues;
import org.sonarqube.ws.client.issues.SearchRequest;

import static com.sonar.python.it.TestsUtils.newWsClient;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

public class TestRulesTest {

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension orchestrator = TestsUtils.dynamicOrchestrator;
  private static final String PROJECT_KEY = "test-rules";
  private static final String PROJECT_NAME = "Test Rules";

  @BeforeAll
  static void prepare() {
    orchestrator.getServer().provisionProject(PROJECT_KEY, PROJECT_NAME);
    orchestrator.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "python-test-rules-profile");
  }

  private SonarScanner newBuild() {
    return orchestrator.createSonarScanner()
      .setProjectDir(new File("projects/test_code"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_NAME)
      .setProjectVersion("1.0");
  }

  @Test
  void test_rules_run_on_test_files() {
    orchestrator.executeBuild(newBuild()
      .setSourceDirs("src")
      .setTestDirs("tests"));

    List<Issues.Issue> assertOnTupleIssues = issues("python:S5905");
    List<Issues.Issue> dedicatedAssertionIssues = issues("python:S5906");
    List<Issues.Issue> mainRuleIssues = issues("python:S3923");
    assertThat(assertOnTupleIssues).hasSize(2);
    assertThat(dedicatedAssertionIssues).hasSize(1);
    assertThat(mainRuleIssues).hasSize(1);
    assertIssue(assertOnTupleIssues.get(0), 2, "Fix this assertion on a tuple literal.", "test-rules:src/some_code.py");
    assertIssue(assertOnTupleIssues.get(1), 3, "Fix this assertion on a tuple literal.", "test-rules:tests/test_my_code.py");
    assertIssue(dedicatedAssertionIssues.get(0), 14, "Consider using \"assertEqual\" instead.", "test-rules:tests/test_my_code.py");
    assertIssue(mainRuleIssues.get(0), 3, "Remove this if statement or edit its code blocks so that they're not all the same.", "test-rules:src/some_code.py");
  }

  @Test
  void main_rules_suppressed_when_all_files_declared_as_test() {
    // main_src/ is an empty placeholder so sonar.sources points to a valid path with no Python files.
    // All actual Python files are under sonar.tests=src,tests → classified as TEST → S3923 (MAIN-scoped) fires nowhere.
    orchestrator.executeBuild(newBuild()
      .setSourceDirs("main_src")
      .setTestDirs("src,tests"));

    List<Issues.Issue> testRuleIssues = issues("python:S5905");
    List<Issues.Issue> mainRuleIssues = issues("python:S3923");

    assertThat(testRuleIssues).hasSize(2);
    assertThat(mainRuleIssues).isEmpty();
  }

  @Test
  void auto_detect_test_files_when_sonar_tests_not_configured() {
    // When sonar.tests is not set, all files get InputFile.Type.MAIN from the platform.
    // The plugin's path-based heuristic detects files in the `tests/` directory and
    // suppresses MAIN-scoped rules on them, while CheckScope.ALL and TEST-scoped rules still fire.
    orchestrator.executeBuild(newBuild()
      .setProperty("sonar.sources", "src,tests"));  // all files are MAIN — no sonar.tests configured

    List<Issues.Issue> mainRuleIssues = issues("python:S3923");
    List<Issues.Issue> allFilesRuleIssues = issues("python:S5905");
    List<Issues.Issue> testScopedRuleIssues = issues("python:S5906");

    // S3923 (MAIN-scoped): fires only on src/some_code.py; heuristic silences it on tests/
    assertThat(mainRuleIssues).hasSize(1);
    assertIssue(mainRuleIssues.get(0), 3,
      "Remove this if statement or edit its code blocks so that they're not all the same.",
      "test-rules:src/some_code.py");

    // S5905 (CheckScope.ALL): fires on both files regardless of effective type
    assertThat(allFilesRuleIssues).hasSize(2);
    assertIssue(allFilesRuleIssues.get(0), 2,
      "Fix this assertion on a tuple literal.", "test-rules:src/some_code.py");
    assertIssue(allFilesRuleIssues.get(1), 3,
      "Fix this assertion on a tuple literal.", "test-rules:tests/test_my_code.py");

    // S5906 (TEST-scoped): fires on heuristic-detected test file
    assertThat(testScopedRuleIssues).hasSize(1);
    assertIssue(testScopedRuleIssues.get(0), 14,
      "Consider using \"assertEqual\" instead.", "test-rules:tests/test_my_code.py");
  }

  @Test
  void sonar_tests_configured_bypasses_heuristic() {
    // When sonar.tests is explicitly configured, the path-based heuristic is inactive.
    // Platform typing governs: files under tests/ are TEST because sonar.tests=tests.
    // Setting sonar.tests is the recommended approach — it gives explicit, predictable
    // control and produces the same isolation as the heuristic.
    orchestrator.executeBuild(newBuild()
      .setProperty("sonar.sources", "src")
      .setProperty("sonar.tests", "tests"));

    List<Issues.Issue> mainRuleIssues = issues("python:S3923");
    List<Issues.Issue> allFilesRuleIssues = issues("python:S5905");
    List<Issues.Issue> testScopedRuleIssues = issues("python:S5906");

    // S3923 (MAIN-scoped): tests/ files are TEST by platform typing — same suppression as heuristic
    assertThat(mainRuleIssues).hasSize(1);
    assertIssue(mainRuleIssues.get(0), 3,
      "Remove this if statement or edit its code blocks so that they're not all the same.",
      "test-rules:src/some_code.py");

    assertThat(allFilesRuleIssues).hasSize(2);
    assertIssue(allFilesRuleIssues.get(0), 2,
      "Fix this assertion on a tuple literal.", "test-rules:src/some_code.py");
    assertIssue(allFilesRuleIssues.get(1), 3,
      "Fix this assertion on a tuple literal.", "test-rules:tests/test_my_code.py");

    // S5906 (TEST-scoped): fires on the explicit test file
    assertThat(testScopedRuleIssues).hasSize(1);
    assertIssue(testScopedRuleIssues.get(0), 14,
      "Consider using \"assertEqual\" instead.", "test-rules:tests/test_my_code.py");
  }

  private void assertIssue(Issues.Issue issue, int expectedLine, String expectedMessage, String expectedComponent) {
    assertThat(issue.getLine()).isEqualTo(expectedLine);
    assertThat(issue.getMessage()).isEqualTo(expectedMessage);
    assertThat(issue.getComponent()).isEqualTo(expectedComponent);
  }

  private static List<Issues.Issue> issues(String rulekey) {
    return newWsClient().issues().search(new SearchRequest().setRules(singletonList(rulekey))).getIssuesList()
      .stream()
      .sorted(Comparator.comparing(Issues.Issue::getComponent).thenComparingInt(Issues.Issue::getLine))
      .toList();
  }
}
