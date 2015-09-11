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

import com.google.common.collect.ImmutableMap;
import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarRunner;
import org.junit.ClassRule;
import org.junit.Test;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static com.sonar.python.it.plugin.Tests.getProjectMeasure;
import static org.fest.assertions.Assertions.assertThat;

public class CoverageTest {

  @ClassRule
  public static Orchestrator orchestrator = Tests.ORCHESTRATOR;

  private static final String PROJECT_KEY = "coverage_project";

  @Test
  public void basic_coverage_reports_with_unix_paths() {
    basic_coverage_reports("ut-coverage.xml");
  }

  @Test
  public void basic_coverage_reports_with_windows_paths() {
    basic_coverage_reports("ut-coverage-windowspaths.xml");
  }

  private void basic_coverage_reports(String utReportPath) {
    SonarRunner build = SonarRunner.create()
      .setProjectDir(new File("projects/coverage_project"))
      .setProperty("sonar.python.coverage.reportPath", utReportPath)
      .setProperty("sonar.python.coverage.itReportPath", "it-coverage.xml")
      .setProperty("sonar.python.coverage.overallReportPath", "it-coverage.xml");
    orchestrator.executeBuild(build);

    Tests.assertProjectMeasures(PROJECT_KEY,
      new ImmutableMap.Builder<String, Integer>()
        .put("lines_to_cover", 8)
        .put("coverage", 80)
        .put("line_coverage", 75)
        .put("branch_coverage", 100)
        .put("it_coverage", 40)
        .put("it_line_coverage", 50)
        .put("it_branch_coverage", 0)
        .put("overall_coverage", 40)
        .put("overall_line_coverage", 50)
        .put("overall_branch_coverage", 0)
        .build());
  }

  @Test
  public void force_zero_coverage_for_untouched_files() throws Exception {
    SonarRunner build = SonarRunner.create()
      .setProjectDir(new File("projects/coverage_project"))
      .setProperty("sonar.python.coverage.reportPath", "ut-coverage.xml")
      .setProperty("sonar.python.coverage.itReportPath", "it-coverage.xml")
      .setProperty("sonar.python.coverage.overallReportPath", "it-coverage.xml")
      .setProperty("sonar.python.coverage.forceZeroCoverage", "true");
    orchestrator.executeBuild(build);

    Tests.assertProjectMeasures(PROJECT_KEY,
      new ImmutableMap.Builder<String, Integer>()
        .put("coverage", 50)
        .put("line_coverage", 42)
        .put("branch_coverage", 100)
        .put("it_coverage", 25)
        .put("it_line_coverage", 28)
        .put("it_branch_coverage", 0)
        .put("overall_coverage", 25)
        .put("overall_line_coverage", 28)
        .put("overall_branch_coverage", 0)
        .build());
  }

  @Test
  public void force_zero_coverage_with_no_report() throws Exception {
    SonarRunner build = SonarRunner.create()
      .setProjectDir(new File("projects/coverage_project"))
      .setProperty("sonar.python.coverage.forceZeroCoverage", "true");
    orchestrator.executeBuild(build);

    Map<String, Integer> expected = new HashMap<>();
    expected.put("coverage", 0);
    expected.put("line_coverage", 0);
    expected.put("branch_coverage", null);
    expected.put("it_coverage", 0);
    expected.put("it_line_coverage", 0);
    expected.put("it_branch_coverage", null);
    expected.put("overall_coverage", 0);
    expected.put("overall_line_coverage", 0);
    expected.put("overall_branch_coverage", null);
    Tests.assertProjectMeasures(PROJECT_KEY, expected);
  }

  @Test
  public void empty_coverage_report() throws Exception {
    SonarRunner build = SonarRunner.create()
      .setProjectDir(new File("projects/coverage_project"))
      .setProperty("sonar.python.coverage.reportPath", "empty.xml")
      .setProperty("sonar.python.coverage.itReportPath", "empty.xml")
      .setProperty("sonar.python.coverage.overallReportPath", "empty.xml");
    BuildResult buildResult = orchestrator.executeBuild(build);

    assertThat(buildResult.getLogs()).matches(".*The report '[^']*' seems to be empty, ignoring.*");
    assertThat(getProjectMeasure(PROJECT_KEY, "coverage")).isNull();
    assertThat(getProjectMeasure(PROJECT_KEY, "it_coverage")).isNull();
    assertThat(getProjectMeasure(PROJECT_KEY, "overall_coverage")).isNull();
  }

}
