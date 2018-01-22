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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import org.junit.ClassRule;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

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
    SonarScanner build = SonarScanner.create()
      .setProjectDir(new File("projects/coverage_project"))
      .setProperty("sonar.python.coverage.reportPath", utReportPath)
      .setProperty("sonar.python.coverage.itReportPath", "it-coverage.xml")
      .setProperty("sonar.python.coverage.overallReportPath", "it-coverage.xml");
    orchestrator.executeBuild(build);

    ImmutableMap<String, Integer> values = new Builder<String, Integer>()
        .put("lines_to_cover", 14)
        .put("coverage", 56)
        .put("line_coverage", 50)
        .put("branch_coverage", 100)
        .build();

    Tests.assertProjectMeasures(PROJECT_KEY, values);
  }

  @Test
  public void no_report() throws Exception {
    SonarScanner build = SonarScanner.create()
      .setProjectDir(new File("projects/coverage_project"));
    orchestrator.executeBuild(build);

    Map<String, Integer> expected = new HashMap<>();
    expected.put("lines_to_cover", 14);
    expected.put("coverage", 0);
    expected.put("line_coverage", 0);
    expected.put("branch_coverage", null);
    Tests.assertProjectMeasures(PROJECT_KEY, expected);
  }

  @Test
  public void empty_coverage_report() throws Exception {
    SonarScanner build = SonarScanner.create()
      .setProjectDir(new File("projects/coverage_project"))
      .setProperty("sonar.python.coverage.reportPath", "empty.xml")
      .setProperty("sonar.python.coverage.itReportPath", "empty.xml")
      .setProperty("sonar.python.coverage.overallReportPath", "empty.xml");
    BuildResult buildResult = orchestrator.executeBuild(build);

    int nbLog = 0;
    for (String s : buildResult.getLogs().split("[\\r\\n]+")) {
      if (s.matches(".*The report '[^']*' seems to be empty, ignoring.*")) {
        nbLog++;
      }
    }
    assertThat(nbLog).isEqualTo(1);
    assertThat(Tests.getMeasureAsDouble(PROJECT_KEY, "coverage")).isZero();
  }

}
