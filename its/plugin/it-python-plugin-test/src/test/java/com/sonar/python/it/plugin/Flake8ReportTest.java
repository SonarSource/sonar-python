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

import java.io.File;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.sonarqube.ws.Common;
import org.sonarqube.ws.Issues;

import static com.sonar.python.it.plugin.TestsUtils.issues;
import static org.assertj.core.api.Assertions.assertThat;

class Flake8ReportTest {

  private static final String PROJECT = "flake8_project";

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension ORCHESTRATOR = TestsUtils.ORCHESTRATOR;

  @Test
  void import_report() {
    ORCHESTRATOR.getServer().provisionProject(PROJECT, PROJECT);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT, "py", "no_rule");
    ORCHESTRATOR.executeBuild(
      ORCHESTRATOR.createSonarScanner()
        .setProjectDir(new File("projects/flake8_project")));

    List<Issues.Issue> issues = issues(PROJECT);
    assertThat(issues).hasSize(5);
    Issues.Issue issue = issues.get(0);
    assertThat(issue.getComponent()).isEqualTo("flake8_project:src/file1.py");
    assertThat(issue.getRule()).isEqualTo("external_flake8:E302");
    assertThat(issue.getMessage()).isEqualTo("expected 2 blank lines, found 1");
    assertThat(issue.getType()).isEqualTo(Common.RuleType.CODE_SMELL);
    assertThat(issue.getSeverity()).isEqualTo(Common.Severity.MAJOR);
    assertThat(issue.getEffort()).isEqualTo("5min");

    // Issue for which we don't have metadata
    issue = issues.get(4);
    assertThat(issue.getComponent()).isEqualTo("flake8_project:src/file1.py");
    assertThat(issue.getRule()).isEqualTo("external_flake8:C901");
    assertThat(issue.getMessage()).isEqualTo("'bar' is too complex (6)");
    assertThat(issue.getType()).isEqualTo(Common.RuleType.CODE_SMELL);
    assertThat(issue.getSeverity()).isEqualTo(Common.Severity.MAJOR);
    assertThat(issue.getEffort()).isEqualTo("5min");
  }

}
