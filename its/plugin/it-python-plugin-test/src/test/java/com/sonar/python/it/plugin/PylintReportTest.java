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

import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.junit5.OrchestratorExtension;
import java.io.File;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import static com.sonar.python.it.plugin.TestsUtils.issues;
import static org.assertj.core.api.Assertions.assertThat;

class PylintReportTest {

  private static final String DEFAULT_PROPERTY = "sonar.python.pylint.reportPaths";
  private static final String LEGACY_PROPERTY = "sonar.python.pylint.reportPath";

  @RegisterExtension
  public static final OrchestratorExtension ORCHESTRATOR = TestsUtils.ORCHESTRATOR;

  @Test
  void import_report() {
    final String projectKey = "pylint_project";
    analyseProjectWithReport(projectKey, DEFAULT_PROPERTY, "pylint-report.txt");
    assertThat(issues(projectKey)).hasSize(4);
  }

  @Test
  void import_report_legacy_key() {
    final String projectKey = "pylint_project_legacy_key";
    analyseProjectWithReport(projectKey, LEGACY_PROPERTY, "pylint-report.txt");
    assertThat(issues(projectKey)).hasSize(4);
  }

  @Test
  void missing_report() {
    final String projectKey = "pylint_project_missing_report";
    analyseProjectWithReport(projectKey, DEFAULT_PROPERTY, "missing");
    assertThat(issues(projectKey)).isEmpty();
  }

  @Test
  void invalid_report() {
    final String projectKey = "pylint_project_invalid_report";
    BuildResult result = analyseProjectWithReport(projectKey, DEFAULT_PROPERTY, "invalid.txt");
    assertThat(result.getLogs()).contains("Cannot parse the line: trash");
    assertThat(issues(projectKey)).isEmpty();
  }

  @Test
  void unknown_rule() {
    final String projectKey = "pylint_project_unknown_rule";
    analyseProjectWithReport(projectKey, DEFAULT_PROPERTY, "rule-unknown.txt");
    assertThat(issues(projectKey)).hasSize(4);
  }

  @Test
  void multiple_reports() {
    final String projectKey = "pylint_project_multiple_reports";
    analyseProjectWithReport(projectKey, DEFAULT_PROPERTY, "pylint-report.txt, rule-unknown.txt");
    assertThat(issues(projectKey)).hasSize(8);
  }

  private static BuildResult analyseProjectWithReport(String projectKey, String property, String reportPaths) {
    ORCHESTRATOR.getServer().provisionProject(projectKey, projectKey);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(projectKey, "py", "no_rule");

    return ORCHESTRATOR.executeBuild(
      SonarScanner.create()
        .setDebugLogs(true)
        .setProjectKey(projectKey)
        .setProjectName(projectKey)
        .setProjectDir(new File("projects/pylint_project"))
        .setProperty(property, reportPaths));
  }

}
