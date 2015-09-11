/*
 * Python :: ITs :: Plugin
 * Copyright (C) 2012 SonarSource and Waleri Enns
 * sonarqube@googlegroups.com
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package com.sonar.python.it.plugin;

import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarRunner;
import org.junit.ClassRule;
import org.junit.Test;
import org.sonar.wsclient.SonarClient;
import org.sonar.wsclient.issue.Issue;
import org.sonar.wsclient.issue.IssueQuery;

import java.io.File;
import java.util.List;

import static org.fest.assertions.Assertions.assertThat;

public class PylintReportTest {

  private static final String PROJECT = "pylint_project";

  @ClassRule
  public static Orchestrator orchestrator = Tests.ORCHESTRATOR;

  private SonarClient wsClient = orchestrator.getServer().wsClient();

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
    assertThat(result.getLogs()).matches(".*DEBUG - Cannot parse the line: trash.*");
    assertThat(issues()).hasSize(0);
  }

  @Test
  public void unknown_rule() throws Exception {
    BuildResult result = analyseProjectWithReport("rule-unknown.txt");
    assertThat(result.getLogs()).matches(".*WARN  - Pylint rule 'C9999' is unknown.*");
    assertThat(issues()).hasSize(0);
  }

  private List<Issue> issues() {
    return wsClient.issueClient().find(IssueQuery.create().componentRoots(PROJECT)).list();
  }

  private BuildResult analyseProjectWithReport(String reportPath) {
    orchestrator.resetData();
    return orchestrator.executeBuild(
      SonarRunner.create()
        .setDebugLogs(true)
        .setProjectDir(new File("projects/pylint_project"))
        .setProfile("pylint-rules")
        .setProperty("sonar.python.pylint.reportPath", reportPath));
  }

}
