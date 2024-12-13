/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
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
package com.sonar.python.it.plugin;

import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.sonarqube.ws.Issues;
import org.sonarqube.ws.client.issues.SearchRequest;

import static com.sonar.python.it.plugin.TestsUtils.newWsClient;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

class TestRulesTest {

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension orchestrator = TestsUtils.ORCHESTRATOR;
  private static final String PROJECT_KEY = "test-rules";
  private static final String PROJECT_NAME = "Test Rules";

  private static SonarScanner BUILD;

  @BeforeAll
  static void prepare() {
    orchestrator.getServer().provisionProject(PROJECT_KEY, PROJECT_NAME);
    orchestrator.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "python-test-rules-profile");
    BUILD = orchestrator.createSonarScanner()
      .setProjectDir(new File("projects/test_code"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_NAME)
      .setProjectVersion("1.0");
  }

  @Test
  void test_rules_run_on_test_files() {
    BUILD.setSourceDirs("src")
      .setTestDirs("tests");
    orchestrator.executeBuild(BUILD);

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
  void declare_all_files_as_test_files() {
    BUILD.setTestDirs(".");
    orchestrator.executeBuild(BUILD);

    List<Issues.Issue> testRuleIssues = issues("python:S5905");
    List<Issues.Issue> mainRuleIssues = issues("python:S3923");

    assertThat(testRuleIssues).hasSize(2);
    assertThat(mainRuleIssues).isEmpty();
  }

  private void assertIssue(Issues.Issue issue, int expectedLine, String expectedMessage, String expectedComponent) {
    assertThat(issue.getLine()).isEqualTo(expectedLine);
    assertThat(issue.getMessage()).isEqualTo(expectedMessage);
    assertThat(issue.getComponent()).isEqualTo(expectedComponent);
  }

  private static List<Issues.Issue> issues(String rulekey) {
    return newWsClient().issues().search(new SearchRequest().setRules(singletonList(rulekey))).getIssuesList();
  }
}
