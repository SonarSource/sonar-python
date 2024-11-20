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
import com.sonar.orchestrator.junit5.OrchestratorExtension;
import java.io.File;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.sonarqube.ws.Common;
import org.sonarqube.ws.Issues;

import static com.sonar.python.it.plugin.TestsUtils.issues;
import static org.assertj.core.api.Assertions.assertThat;

class MypyReportTest {

  private static final String PROJECT = "mypy_project";

  @RegisterExtension
  public static final OrchestratorExtension ORCHESTRATOR = TestsUtils.ORCHESTRATOR;

  @Test
  void import_report() {
    ORCHESTRATOR.getServer().provisionProject(PROJECT, PROJECT);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT, "py", "no_rule");
    ORCHESTRATOR.executeBuild(
      SonarScanner.create()
        .setProjectDir(new File("projects/mypy_project")));

    List<Issues.Issue> issues = issues(PROJECT);
    assertThat(issues).hasSize(5);
    assertIssue(issues.get(0), "external_mypy:arg-type", "Argument 1 to \"greet_all\" has incompatible type \"List[int]\"; expected \"List[str]\"");
    assertIssue(issues.get(1), "external_mypy:no-untyped-def", "Function is missing a type annotation");
    assertIssue(issues.get(2), "external_mypy:import", "Cannot find implementation or library stub for module named \"unknown\"");
    assertIssue(issues.get(3), "external_mypy:no-untyped-call", "Call to untyped function \"no_type_hints\" in typed context");
    assertIssue(issues.get(4), "external_mypy:unknown_mypy_rule", "Unused \"type: ignore\" comment");
  }

  private static void assertIssue(Issues.Issue issue, String rule, String message) {
    assertThat(issue.getComponent()).isEqualTo("mypy_project:src/type_hints_noncompliant.py");
    assertThat(issue.getRule()).isEqualTo(rule);
    assertThat(issue.getMessage()).isEqualTo(message);
    assertThat(issue.getType()).isEqualTo(Common.RuleType.CODE_SMELL);
    assertThat(issue.getSeverity()).isEqualTo(Common.Severity.MAJOR);
    assertThat(issue.getEffort()).isEqualTo("5min");
  }
}
