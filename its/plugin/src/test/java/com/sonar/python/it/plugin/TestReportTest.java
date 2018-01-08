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

  @ClassRule
  public static Orchestrator orchestrator = Tests.ORCHESTRATOR;

  @Before
  public void before() {
    orchestrator.resetData();
  }

  private SonarScanner createBuild(String testReportPath) {
    return SonarScanner.create()
      .setProjectDir(new File("projects/nosetests_project"))
      .setProperty("sonar.python.xunit.reportPath", testReportPath);
  }

  @Test
  public void import_report() throws Exception {
    // sonar.python.xunit.skipDetails=false by default
    orchestrator.executeBuild(createBuild("nosetests.xml"));
    assertProjectMeasures(PROJECT, new ImmutableMap.Builder<String, Integer>()
      .put("tests", 3)
      .put("test_failures", 1)
      .put("test_errors", 1)
      .put("skipped_tests", 1)
      .put("test_success_density", 33)
      .put("test_execution_time", 1)
      .build());
  }

  @Test
  public void simple_mode() throws Exception {
    orchestrator.executeBuild(createBuild("nosetests.xml").setProperty("sonar.python.xunit.skipDetails", "true"));
    Map<String, Integer> values = new HashMap<>();
    values.put("tests", 3);
    values.put("test_failures", 1);
    values.put("test_errors", 1);
    values.put("skipped_tests", 1);
    values.put("test_execution_time", 1);
    values.put("test_success_density", null);

    assertProjectMeasures(PROJECT, values);
  }

  @Test
  public void missing_test_report() throws Exception {
    orchestrator.executeBuild(createBuild("missing.xml"));
    assertProjectMeasures(PROJECT, nullMeasures());
  }

  @Test
  public void missing_test_report_with_simple_mode() throws Exception {
    orchestrator.executeBuild(createBuild("missing.xml").setProperty("sonar.python.xunit.skipDetails", "true"));
    assertProjectMeasures(PROJECT, nullMeasures());
  }

  @Test
  public void invalid_test_report() throws Exception {
    BuildResult result = orchestrator.executeBuildQuietly(createBuild("invalid_report.xml"));
    assertThat(result.isSuccess()).isFalse();
    assertThat(result.getLogs()).contains("Cannot feed the data into sonar");
  }

  private Map<String, Integer> nullMeasures() {
    Set<String> metrics = ImmutableSet.of("tests", "test_failures", "test_errors", "skipped_tests", "test_success_density", "test_execution_time");
    return Maps.asMap(metrics, Functions.<Integer>constant(null));
  }

}
