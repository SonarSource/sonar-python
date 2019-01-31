/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2019 SonarSource SA
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

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.junit.Before;
import org.junit.ClassRule;
import org.junit.Test;

import static com.sonar.python.it.plugin.Tests.assertProjectMeasures;
import static org.assertj.core.api.Assertions.assertThat;

public class TestReportTest {

  private static final String PROJECT = "nosetests_project";
  public static final String TESTS = "tests";
  public static final String TEST_FAILURES = "test_failures";
  public static final String TEST_ERRORS = "test_errors";
  public static final String SKIPPED_TESTS = "skipped_tests";
  public static final String TEST_SUCCESS_DENSITY = "test_success_density";
  public static final String TEST_EXECUTION_TIME = "test_execution_time";

  @ClassRule
  public static final Orchestrator ORCHESTRATOR = Tests.ORCHESTRATOR;

  @Before
  public void before() {
    ORCHESTRATOR.resetData();
  }

  private static SonarScanner createBuild(String testReportPath) {
    return SonarScanner.create()
      .setProjectDir(new File("projects/nosetests_project"))
      .setProperty("sonar.python.xunit.reportPath", testReportPath);
  }

  @Test
  public void import_report() {
    // sonar.python.xunit.skipDetails=false by default
    ORCHESTRATOR.executeBuild(createBuild("nosetests.xml"));
    assertProjectMeasures(PROJECT, new ImmutableMap.Builder<String, Integer>()
      .put(TESTS, 3)
      .put(TEST_FAILURES, 1)
      .put(TEST_ERRORS, 1)
      .put(SKIPPED_TESTS, 1)
      .put(TEST_SUCCESS_DENSITY, 33)
      .put(TEST_EXECUTION_TIME, 1)
      .build());
  }

  @Test
  public void import_pytest_report() {
    ORCHESTRATOR.executeBuild(createBuild("pytest.xml"));
    assertProjectMeasures(PROJECT, new ImmutableMap.Builder<String, Integer>()
      .put(TESTS, 3)
      .put(TEST_FAILURES, 2)
      .put(TEST_ERRORS, 0)
      .put(SKIPPED_TESTS, 1)
      .put(TEST_SUCCESS_DENSITY, 33)
      .put(TEST_EXECUTION_TIME, 1)
      .build());
  }

  @Test
  public void simple_mode() {
    ORCHESTRATOR.executeBuild(createBuild("nosetests.xml").setProperty("sonar.python.xunit.skipDetails", "true"));
    Map<String, Integer> values = new HashMap<>();
    values.put(TESTS, 3);
    values.put(TEST_FAILURES, 1);
    values.put(TEST_ERRORS, 1);
    values.put(SKIPPED_TESTS, 1);
    values.put(TEST_EXECUTION_TIME, 1);
    values.put(TEST_SUCCESS_DENSITY, null);

    assertProjectMeasures(PROJECT, values);
  }

  @Test
  public void missing_test_report() {
    ORCHESTRATOR.executeBuild(createBuild("missing.xml"));
    assertProjectMeasures(PROJECT, nullMeasures());
  }

  @Test
  public void missing_test_report_with_simple_mode() {
    ORCHESTRATOR.executeBuild(createBuild("missing.xml").setProperty("sonar.python.xunit.skipDetails", "true"));
    assertProjectMeasures(PROJECT, nullMeasures());
  }

  @Test
  public void invalid_test_report() {
    BuildResult result = ORCHESTRATOR.executeBuildQuietly(createBuild("invalid_report.xml"));
    assertThat(result.isSuccess()).isFalse();
    assertThat(result.getLogs()).contains("Cannot feed the data into sonar");
  }

  private static Map<String, Integer> nullMeasures() {
    Set<String> metrics = ImmutableSet.of(TESTS, TEST_FAILURES, TEST_ERRORS, SKIPPED_TESTS, TEST_SUCCESS_DENSITY, TEST_EXECUTION_TIME);
    return Maps.asMap(metrics, Functions.<Integer>constant(null));
  }

}
