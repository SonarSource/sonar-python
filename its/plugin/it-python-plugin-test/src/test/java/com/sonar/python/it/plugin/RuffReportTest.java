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

import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.sonarqube.ws.Common;
import org.sonarqube.ws.Issues;

import static com.sonar.python.it.plugin.Tests.issues;
import static org.assertj.core.api.Assertions.assertThat;

class RuffReportTest {

  private final String PROJECT = "ruff_project";

  @RegisterExtension
  public static final Orchestrator ORCHESTRATOR = Tests.ORCHESTRATOR;

  @Test
  void import_report() {
    ORCHESTRATOR.getServer().provisionProject(PROJECT, PROJECT);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT, "py", "no_rule");
    ORCHESTRATOR.executeBuild(
        SonarScanner.create()
            .setProjectDir(new File("projects/ruff_project")));

    List<Issues.Issue> issues = issues(PROJECT).stream().sorted(Comparator.comparing(Issues.Issue::getRule))
        .collect(Collectors.toList());
    assertThat(issues).hasSize(2);

    Issues.Issue firstIssue = issues.get(0);
    assertThat(firstIssue.getComponent()).isEqualTo("ruff_project:src/file1.py");
    assertThat(firstIssue.getRule()).isEqualTo("external_ruff:E501");
    assertThat(firstIssue.getMessage()).isEqualTo("Line too long (108 > 88 characters)");
    assertThat(firstIssue.getType()).isEqualTo(Common.RuleType.CODE_SMELL);
    assertThat(firstIssue.getSeverity()).isEqualTo(Common.Severity.MAJOR);
    assertThat(firstIssue.getEffort()).isEqualTo("5min");

    Issues.Issue secondIssue = issues.get(1);
    assertThat(secondIssue.getComponent()).isEqualTo("ruff_project:src/file1.py");
    assertThat(secondIssue.getRule()).isEqualTo("external_ruff:F821");
    assertThat(secondIssue.getMessage()).isEqualTo("Undefined name `random`");
    assertThat(secondIssue.getType()).isEqualTo(Common.RuleType.CODE_SMELL);
    assertThat(secondIssue.getSeverity()).isEqualTo(Common.Severity.MAJOR);
    assertThat(secondIssue.getEffort()).isEqualTo("5min");

  }

}
