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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import static org.assertj.core.api.Assertions.assertThat;

class CoverageTest {

  private static final String COVERAGE_PROJECT = "projects/coverage_project";
  @RegisterExtension
  public static final ConcurrentOrchestratorExtension ORCHESTRATOR = TestsUtils.ORCHESTRATOR;

  private static final String PROJECT_KEY = "coverage_project";
  private static final String LINES_TO_COVER = "lines_to_cover";
  private static final String COVERAGE = "coverage";
  private static final String LINE_COVERAGE = "line_coverage";
  private static final String BRANCH_COVERAGE = "branch_coverage";
  private static final String EMPTY_XML = "empty.xml";
  private static final String DEPRECATED_COVERAGE_REPORT_PATH = "sonar.python.coverage.reportPath";
  private static final String COVERAGE_REPORT_PATHS = "sonar.python.coverage.reportPaths";

  @Test
  void basic_coverage_reports_with_unix_paths() {
    basicCoverageReports("ut-coverage.xml");
  }

  @Test
  void basic_coverage_reports_with_windows_paths() {
    basicCoverageReports("ut-coverage-windowspaths.xml");
  }

  private static void basicCoverageReports(String utReportPath) {
    SonarScanner build = ORCHESTRATOR.createSonarScanner()
      .setProjectDir(new File(COVERAGE_PROJECT))
      .setProperty(DEPRECATED_COVERAGE_REPORT_PATH, "someReport")
      .setProperty(COVERAGE_REPORT_PATHS, utReportPath+",it-coverage1.xml,it-coverage2.xml");

    BuildResult result = ORCHESTRATOR.executeBuild(build);

    ImmutableMap<String, Integer> values = new Builder<String, Integer>()
        .put(LINES_TO_COVER, 14)
        .put(COVERAGE, 56)
        .put(LINE_COVERAGE, 50)
        .put(BRANCH_COVERAGE, 100)
        .build();

    TestsUtils.assertProjectMeasures(PROJECT_KEY, values);
    assertThat(result.getLogs()).contains("Property 'sonar.python.coverage.reportPath' has been removed. Please use 'sonar.python.coverage.reportPaths' instead.");
  }

  @Test
  void default_values() {
    SonarScanner build = ORCHESTRATOR.createSonarScanner()
      .setProjectDir(new File(COVERAGE_PROJECT));
    ORCHESTRATOR.executeBuild(build);

    ImmutableMap<String, Integer> values = new Builder<String, Integer>()
      .put(LINES_TO_COVER, 14)
      .put(COVERAGE, 25)
      .put(LINE_COVERAGE, 28)
      .put(BRANCH_COVERAGE, 0)
      .build();
    TestsUtils.assertProjectMeasures(PROJECT_KEY, values);
  }

  @Test
  void empty_property() {
    SonarScanner build = ORCHESTRATOR.createSonarScanner()
      .setProjectDir(new File(COVERAGE_PROJECT))
      .setProperty(DEPRECATED_COVERAGE_REPORT_PATH, "")
      .setProperty(COVERAGE_REPORT_PATHS, "");
    ORCHESTRATOR.executeBuild(build);

    Map<String, Integer> expected = new HashMap<>();
    expected.put(LINES_TO_COVER, 14);
    expected.put(COVERAGE, 0);
    expected.put(LINE_COVERAGE, 0);
    expected.put(BRANCH_COVERAGE, null);
    TestsUtils.assertProjectMeasures(PROJECT_KEY, expected);
  }

  @Test
  void empty_coverage_report() {
    SonarScanner build = ORCHESTRATOR.createSonarScanner()
      .setProjectDir(new File(COVERAGE_PROJECT))
      .setProperty(DEPRECATED_COVERAGE_REPORT_PATH, EMPTY_XML)
      .setProperty(COVERAGE_REPORT_PATHS, EMPTY_XML);
    BuildResult buildResult = ORCHESTRATOR.executeBuild(build);

    int nbLog = 0;
    for (String s : buildResult.getLogs().split("[\\r\\n]+")) {
      if (s.matches(".*The report '[^']*' seems to be empty, ignoring.*")) {
        nbLog++;
      }
    }
    assertThat(nbLog).isEqualTo(1);
    assertThat(TestsUtils.getMeasureAsDouble(PROJECT_KEY, COVERAGE)).isZero();
  }

}
