/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2018 SonarSource SA
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
import org.sonarqube.ws.client.issue.SearchWsRequest;

import static com.sonar.python.it.plugin.Tests.newWsClient;
import static java.util.Collections.singletonList;
import static org.assertj.core.api.Assertions.assertThat;

public class PylintReportTest {

  private static final String PROJECT = "pylint_project";

  @ClassRule
  public static Orchestrator orchestrator = Tests.ORCHESTRATOR;

  @Test
  public void import_report() throws Exception {
    analyseProjectWithReport("pylint-report.txt");
    assertThat(issues()).hasSize(4);
  }

  @Test
  public void missing_report() throws Exception {
    analyseProjectWithReport("missing");
    assertThat(issues()).hasSize(0);
  }

  @Test
  public void invalid_report() throws Exception {
    BuildResult result = analyseProjectWithReport("invalid.txt");
    assertThat(result.getLogs()).contains("Cannot parse the line: trash");
    assertThat(issues()).hasSize(0);
  }

  @Test
  public void unknown_rule() throws Exception {
    BuildResult result = analyseProjectWithReport("rule-unknown.txt");
    assertThat(result.getLogs()).contains("Pylint rule 'C9999' is unknown");
    assertThat(issues()).hasSize(0);
  }

  private List<Issue> issues() {
    return newWsClient().issues().search(new SearchWsRequest().setProjects(singletonList(PROJECT))).getIssuesList();
  }

  private BuildResult analyseProjectWithReport(String reportPath) {
    orchestrator.resetData();
    orchestrator.getServer().provisionProject(PROJECT, PROJECT);
    orchestrator.getServer().associateProjectToQualityProfile(PROJECT, "py", "pylint-rules");
    return orchestrator.executeBuild(
      SonarScanner.create()
        .setDebugLogs(true)
        .setProjectDir(new File("projects/pylint_project"))
        .setProperty("sonar.python.pylint.reportPath", reportPath));
  }

}
