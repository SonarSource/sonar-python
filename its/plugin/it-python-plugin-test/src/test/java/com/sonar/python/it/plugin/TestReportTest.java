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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import static com.sonar.python.it.plugin.TestsUtils.assertProjectMeasures;
import static org.assertj.core.api.Assertions.assertThat;

class TestReportTest {

  public static final String TESTS = "tests";
  public static final String TEST_FAILURES = "test_failures";
  public static final String TEST_ERRORS = "test_errors";
  public static final String SKIPPED_TESTS = "skipped_tests";
  public static final String TEST_SUCCESS_DENSITY = "test_success_density";
  public static final String TEST_EXECUTION_TIME = "test_execution_time";

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension ORCHESTRATOR = TestsUtils.ORCHESTRATOR;

  private static SonarScanner createBuild(String projectKey, String testReportPath) {
    return ORCHESTRATOR.createSonarScanner()
      .setProjectKey(projectKey)
      .setProjectName(projectKey)
      .setProjectDir(new File("projects/nosetests_project"))
      .setProperty("sonar.python.xunit.reportPath", testReportPath);
  }

  @Test
  void import_report() {
    final String projectKey = "nosetests_project";
    // sonar.python.xunit.skipDetails=false by default
    ORCHESTRATOR.executeBuild(createBuild(projectKey, "nosetests.xml"));
    assertProjectMeasures(projectKey, new ImmutableMap.Builder<String, Integer>()
      .put(TESTS, 3)
      .put(TEST_FAILURES, 1)
      .put(TEST_ERRORS, 1)
      .put(SKIPPED_TESTS, 1)
      .put(TEST_SUCCESS_DENSITY, 33)
      .put(TEST_EXECUTION_TIME, 1)
      .build());
  }

  @Test
  void import_pytest_report() {
    final String projectKey = "pytest";
    ORCHESTRATOR.executeBuild(createBuild(projectKey, "pytest.xml"));
    assertProjectMeasures(projectKey, new ImmutableMap.Builder<String, Integer>()
      .put(TESTS, 3)
      .put(TEST_FAILURES, 2)
      .put(TEST_ERRORS, 0)
      .put(SKIPPED_TESTS, 1)
      .put(TEST_SUCCESS_DENSITY, 33)
      .put(TEST_EXECUTION_TIME, 1)
      .build());
  }

  @Test
  void simple_mode() {
    final String projectKey = "nosetests_simple";
    ORCHESTRATOR.executeBuild(createBuild(projectKey, "nosetests.xml").setProperty("sonar.python.xunit.skipDetails", "true"));
    Map<String, Integer> values = new HashMap<>();
    values.put(TESTS, 3);
    values.put(TEST_FAILURES, 1);
    values.put(TEST_ERRORS, 1);
    values.put(SKIPPED_TESTS, 1);
    values.put(TEST_EXECUTION_TIME, 1);
    values.put(TEST_SUCCESS_DENSITY, null);

    assertProjectMeasures(projectKey, values);
  }

  @Test
  void missing_test_report() {
    final String projectKey = "nosetests_missing";
    ORCHESTRATOR.executeBuild(createBuild(projectKey, "missing.xml"));
    assertProjectMeasures(projectKey, nullMeasures());
  }

  @Test
  void missing_test_report_with_simple_mode() {
    final String projectKey = "nosetests_missing_simple";
    ORCHESTRATOR.executeBuild(createBuild(projectKey, "missing.xml").setProperty("sonar.python.xunit.skipDetails", "true"));
    assertProjectMeasures(projectKey, nullMeasures());
  }

  @Test
  void invalid_test_report() {
    final String projectKey = "nosetests_invalid";
    BuildResult result = ORCHESTRATOR.executeBuildQuietly(createBuild(projectKey, "invalid_report.xml"));
    assertThat(result.isSuccess()).isTrue();
    assertThat(result.getLogs()).contains("Cannot read report 'invalid_report.xml', the following exception occurred:" +
      " Unexpected character 't' (code 116) in prolog; expected '<'\n" +
      " at [row,col {unknown-source}]: [1,1]");
  }

  private static Map<String, Integer> nullMeasures() {
    Set<String> metrics = ImmutableSet.of(TESTS, TEST_FAILURES, TEST_ERRORS, SKIPPED_TESTS, TEST_SUCCESS_DENSITY, TEST_EXECUTION_TIME);
    return Maps.asMap(metrics, i -> null);
  }

}
