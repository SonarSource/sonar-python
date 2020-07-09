/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2020 SonarSource SA
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
import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.List;
import org.junit.ClassRule;
import org.junit.Test;
import org.sonarqube.ws.Issues.Issue;
import org.sonarqube.ws.client.issues.SearchRequest;

import static com.sonar.python.it.plugin.Tests.newWsClient;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

public class PylintReportTest {

  private static final String PROJECT = "pylint_project";
  private static final String DEFAULT_PROPERTY = "sonar.python.pylint.reportPaths";
  private static final String LEGACY_PROPERTY = "sonar.python.pylint.reportPath";

  @ClassRule
  public static final Orchestrator ORCHESTRATOR = Tests.ORCHESTRATOR;

  @Test
  public void import_report() {
    analyseProjectWithReport(DEFAULT_PROPERTY, "pylint-report.txt");
    assertThat(issues()).hasSize(4);
  }

  @Test
  public void import_report_legacy_key() {
    analyseProjectWithReport(LEGACY_PROPERTY, "pylint-report.txt");
    assertThat(issues()).hasSize(4);
  }

  @Test
  public void missing_report() {
    analyseProjectWithReport(DEFAULT_PROPERTY, "missing");
    assertThat(issues()).isEmpty();
  }

  @Test
  public void invalid_report() {
    BuildResult result = analyseProjectWithReport(DEFAULT_PROPERTY, "invalid.txt");
    assertThat(result.getLogs()).contains("Cannot parse the line: trash");
    assertThat(issues()).isEmpty();
  }

  @Test
  public void unknown_rule() {
    BuildResult result = analyseProjectWithReport(DEFAULT_PROPERTY, "rule-unknown.txt");
    assertThat(issues()).hasSize(4);
  }

  @Test
  public void multiple_reports() {
    analyseProjectWithReport(DEFAULT_PROPERTY, "pylint-report.txt, rule-unknown.txt");
    assertThat(issues()).hasSize(8);
  }

  private static List<Issue> issues() {
    return newWsClient().issues().search(new SearchRequest().setProjects(singletonList(PROJECT))).getIssuesList();
  }

  private static BuildResult analyseProjectWithReport(String property, String reportPaths) {
    ORCHESTRATOR.resetData();
    ORCHESTRATOR.getServer().provisionProject(PROJECT, PROJECT);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT, "py", "no_rule");

    return ORCHESTRATOR.executeBuild(
      SonarScanner.create()
        .setDebugLogs(true)
        .setProjectDir(new File("projects/pylint_project"))
        .setProperty(property, reportPaths));
  }

}
