/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2023 SonarSource SA
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
package com.sonar.python.it.plugin;

import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.junit4.OrchestratorRule;
import java.io.File;
import java.util.List;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.sonarqube.ws.Issues;
import org.sonarqube.ws.client.issues.SearchRequest;

import static com.sonar.python.it.plugin.Tests.newWsClient;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

public class TestRulesTest {

  @ClassRule
  public static OrchestratorRule orchestrator = Tests.ORCHESTRATOR;
  private static final String PROJECT_KEY = "test-rules";
  private static final String PROJECT_NAME = "Test Rules";

  private static SonarScanner BUILD;

  @BeforeClass
  public static void prepare() {
    orchestrator.getServer().provisionProject(PROJECT_KEY, PROJECT_NAME);
    orchestrator.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "python-test-rules-profile");
    BUILD = SonarScanner.create()
      .setProjectDir(new File("projects/test_code"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_NAME)
      .setProjectVersion("1.0");
  }

  @Test
  public void test_rules_run_on_test_files() {
    BUILD.setSourceDirs("src")
      .setTestDirs("tests");
    orchestrator.executeBuild(BUILD);

    List<Issues.Issue> testRuleIssues = issues("python:S5905");
    List<Issues.Issue> mainRuleIssues = issues("python:S3923");
    assertThat(testRuleIssues).hasSize(2);
    assertThat(mainRuleIssues).hasSize(1);
    assertIssue(testRuleIssues.get(0), 2, "Fix this assertion on a tuple literal.", "test-rules:src/some_code.py");
    assertIssue(testRuleIssues.get(1), 3, "Fix this assertion on a tuple literal.", "test-rules:tests/test_my_code.py");
    assertIssue(mainRuleIssues.get(0), 3, "Remove this if statement or edit its code blocks so that they're not all the same.", "test-rules:src/some_code.py");
  }

  @Test
  public void declare_all_files_as_test_files() {
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
