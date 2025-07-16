/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2025 SonarSource SA
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
import com.sonar.python.it.ConcurrentOrchestratorExtension;
import com.sonar.python.it.TestsUtils;
import java.io.File;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.sonarqube.ws.Issues;

import static com.sonar.python.it.TestsUtils.issues;
import static org.assertj.core.api.Assertions.assertThat;

public class NoSonarTest {

  private static final String NO_SONAR_PROJECT_KEY = "nosonar";
  private static final String EXTERNAL_ISSUE_PROJECT_KEY = "external-issues";

  private static final String PROFILE_NAME = "nosonar";

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension ORCHESTRATOR = TestsUtils.dynamicOrchestrator;

  @Test
  void test_externalIssues() {
    SonarScanner build = createScanner(EXTERNAL_ISSUE_PROJECT_KEY, "projects/nosonar/external-issue-project")
      .setProperty("sonar.python.flake8.reportPaths", "flake8-report.txt");
    analyzeProject(build);

    List<Issues.Issue> issues = issues(EXTERNAL_ISSUE_PROJECT_KEY);
    assertThat(issues)
      .hasSize(6)
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 1))
      .anySatisfy(issue -> assertIssueMatches(issue, "external_flake8:E261", 1))

      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 2))
      .anySatisfy(issue -> assertIssueMatches(issue, "external_flake8:E261", 2))

      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 3))
      .anySatisfy(issue -> assertIssueMatches(issue, "external_flake8:E261", 3))
      ;
  }

  @Test
  void test_nosonar() {
    analyzeProject(createScanner(NO_SONAR_PROJECT_KEY, "projects/nosonar/nosonar-project"));

    List<Issues.Issue> issues = issues(NO_SONAR_PROJECT_KEY);
    assertThat(issues)
      .hasSize(19)
      // basic no-sonar examples
      .anySatisfy(issue -> assertIssueMatches(issue, "python:PrintStatementUsage", 1))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 2))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 3))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 4))

      // no-sonar with comments
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 6))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 7))

      // no-sonar with multiple rules
      .anySatisfy(issue -> assertIssueMatches(issue, "python:PrintStatementUsage", 9))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:OneStatementPerLine", 9))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 10))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 11))

      // invalid no-sonar
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 14))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 15))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:PrintStatementUsage", 16))
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 16))

      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 19))

      // no-sonar on the last line
      .anySatisfy(issue -> assertIssueMatches(issue, "python:NoSonar", 20));
  }

  private void analyzeProject(SonarScanner scanner) {
    String projectKey = scanner.getProperty("sonar.projectKey");
    ORCHESTRATOR.getServer().provisionProject(projectKey, projectKey);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(projectKey, "py", PROFILE_NAME);
    ORCHESTRATOR.executeBuild(scanner);
  }

  private SonarScanner createScanner(String projectKey, String projectDir) {
    return ORCHESTRATOR.createSonarScanner()
      .setProjectDir(new File(projectDir))
      .setProjectKey(projectKey)
      .setProjectName(projectKey)
      .setProjectVersion("1.0-SNAPSHOT")
      .setSourceDirs(".");
  }

  private static void assertIssueMatches(Issues.Issue issue, String expectedRule, int expectedLine) {
    assertThat(issue.getRule()).isEqualTo(expectedRule);
    assertThat(issue.getLine()).isEqualTo(expectedLine);
  }
}
