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

import static com.sonar.python.it.TestsUtils.issues;

import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.python.it.ConcurrentOrchestratorExtension;
import com.sonar.python.it.IssueListAssert;
import com.sonar.python.it.TestsUtils;
import java.io.File;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

public class NoSonarTest {

  private static final String NO_SONAR_PROJECT_KEY = "nosonar";
  private static final String EXTERNAL_ISSUE_PROJECT_KEY = "external-issues";
  private static final String NOQA_PROJECT_KEY = "noqa";

  private static final String PROFILE_NAME = "nosonar";

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension ORCHESTRATOR = TestsUtils.dynamicOrchestrator;

  @Test
  void test_externalIssues() {
    SonarScanner build = createScanner(EXTERNAL_ISSUE_PROJECT_KEY, "projects/nosonar/external-issue-project")
        .setProperty("sonar.python.flake8.reportPaths", "flake8-report.txt");
    analyzeProject(build);

    IssueListAssert.assertThat(issues(EXTERNAL_ISSUE_PROJECT_KEY))
        .hasSize(6)
        .containsIssue(1, "external_flake8:E261")
        .containsIssue(1, "python:NoSonar")
        .containsIssue(2, "external_flake8:E261")
        .containsIssue(2, "python:NoSonar")
        .containsIssue(3, "external_flake8:E261")
        .containsIssue(3, "python:NoSonar");
  }

  @Test
  void test_nosonar() {
    analyzeProject(createScanner(NO_SONAR_PROJECT_KEY, "projects/nosonar/nosonar-project"));

    IssueListAssert.assertThat(issues(NO_SONAR_PROJECT_KEY))
        .hasSize(21)
        // basic no-sonar checks
        .containsIssue(1, "python:PrintStatementUsage")
        .containsIssue(2, "python:NoSonar").doesNotContainIssue(2, "python:PrintStatementUsage")
        .containsIssue(3, "python:NoSonar").doesNotContainIssue(3, "python:PrintStatementUsage")
        .containsIssue(4, "python:NoSonar").doesNotContainIssue(4, "python:PrintStatementUsage")

        // no-sonar with comments
        .containsIssue(6, "python:NoSonar").doesNotContainIssue(6, "python:PrintStatementUsage")
        .containsIssue(7, "python:NoSonar").doesNotContainIssue(7, "python:PrintStatementUsage")

        // no-sonar with multiple rules
        .containsIssue(9, "python:OneStatementPerLine")
        .containsIssue(9, "python:PrintStatementUsage")
        .containsIssue(9, "python:PrintStatementUsage")

        .containsIssue(10, "python:NoSonar")
        .doesNotContainIssue(10, "python:PrintStatementUsage")
        .doesNotContainIssue(10, "python:OneStatementPerLine")

        .containsIssue(11, "python:NoSonar")
        .containsIssue(11, "python:OneStatementPerLine")
        .doesNotContainIssue(11, "python:PrintStatementUsage")

        // invalid no-sonar
        .containsIssue(13, "python:NoSonar")
        .containsIssue(14, "python:NoSonar")
        .containsIssue(15, "python:NoSonar")
        .containsIssue(16, "python:NoSonar")
        .containsIssue(16, "python:PrintStatementUsage")

        // no-sonar at the end of file
        .containsIssue(19, "python:NoSonar")
        .containsIssue(20, "python:NoSonar")
        .doesNotContainIssue(20, "python:PrintStatementUsage")
        .doesNotContainIssue(20, "python:OneStatementPerLine");
  }

  @Test
  void test_noqa() {
    analyzeProject(createScanner(NOQA_PROJECT_KEY, "projects/nosonar/noqa-project"));

    IssueListAssert.assertThat(issues(NOQA_PROJECT_KEY))
        .hasSize(19)
        // basic noqa checks
        .containsIssue(1, "python:PrintStatementUsage")
        .containsIssue(2, "python:S1309").doesNotContainIssue(2, "python:PrintStatementUsage")
        .containsIssue(3, "python:S1309").doesNotContainIssue(3, "python:PrintStatementUsage")
        .containsIssue(4, "python:S1309").doesNotContainIssue(4, "python:PrintStatementUsage")
        
        // noqa with specific rules
        .containsIssue(6, "python:S1309")
        .containsIssue(7, "python:S1309").doesNotContainIssue(6, "python:PrintStatementUsage")
        .containsIssue(8, "python:S1309").doesNotContainIssue(7, "python:PrintStatementUsage")
        
        // noqa with multiple rules
        .containsIssue(11, "python:OneStatementPerLine").containsIssue(11, "python:PrintStatementUsage")
        .containsIssue(12, "python:S1309").doesNotContainIssue(12, "python:PrintStatementUsage")
        .doesNotContainIssue(12, "python:OneStatementPerLine")
        .containsIssue(13, "python:S1309").doesNotContainIssue(13, "python:PrintStatementUsage")
        .doesNotContainIssue(13, "python:OneStatementPerLine")
        .containsIssue(14, "python:S1309").doesNotContainIssue(14, "python:PrintStatementUsage")
        .doesNotContainIssue(14, "python:OneStatementPerLine")

        // invalid noqa
        .containsIssue(17, "python:S1309").doesNotContainIssue(17, "python:PrintStatementUsage")
        .containsIssue(18, "python:S1309").containsIssue(18, "python:PrintStatementUsage")
        .containsIssue(19, "python:S1309").doesNotContainIssue(19, "python:PrintStatementUsage").doesNotContainIssue(19, "python:PrintStatementUsage")
        
        // noqa at the end of file
        .containsIssue(21, "python:S1309")
        .containsIssue(22, "python:S1309").doesNotContainIssue(22, "python:PrintStatementUsage");
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
}
