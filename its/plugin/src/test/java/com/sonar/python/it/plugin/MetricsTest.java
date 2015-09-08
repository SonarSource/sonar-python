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
import com.sonar.orchestrator.build.SonarRunner;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.sonar.wsclient.Sonar;
import org.sonar.wsclient.services.Measure;
import org.sonar.wsclient.services.Resource;
import org.sonar.wsclient.services.ResourceQuery;

import java.io.File;

import static org.fest.assertions.Assertions.assertThat;

public class MetricsTest {

  private static final String PROJECT_KEY = "metrics";

  @ClassRule
  public static Orchestrator ORCHESTRATOR = Tests.ORCHESTRATOR;

  private static Sonar wsClient;

  @BeforeClass
  public static void startServer() {
    SonarRunner build = SonarRunner.create()
      .setProjectDir(new File("projects/metrics"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1.0-SNAPSHOT")
      .setProfile("no_rule")
      .setSourceDirs("src");
    ORCHESTRATOR.executeBuild(build);

    wsClient = ORCHESTRATOR.getServer().getWsClient();
  }

  @Test
  public void project_level() {
    // Size
    assertThat(getProjectMeasure("ncloc").getIntValue()).isEqualTo(1);
    assertThat(getProjectMeasure("lines").getIntValue()).isEqualTo(6);
    assertThat(getProjectMeasure("files").getIntValue()).isEqualTo(1);
    assertThat(getProjectMeasure("statements").getIntValue()).isEqualTo(1);
    assertThat(getProjectMeasure("directories").getIntValue()).isEqualTo(1);
    assertThat(getProjectMeasure("functions").getIntValue()).isEqualTo(0);
    assertThat(getProjectMeasure("classes").getIntValue()).isEqualTo(0);
    // Documentation
    assertThat(getProjectMeasure("comment_lines").getIntValue()).isEqualTo(1);
    assertThat(getProjectMeasure("comment_lines_density").getValue()).isEqualTo(50.0);
    // Complexity
    assertThat(getProjectMeasure("complexity").getValue()).isEqualTo(0.0);
    assertThat(getProjectMeasure("function_complexity")).isNull();
    assertThat(getProjectMeasure("function_complexity_distribution").getData()).isEqualTo("1=0;2=0;4=0;6=0;8=0;10=0;12=0;20=0;30=0");
    assertThat(getProjectMeasure("file_complexity").getValue()).isEqualTo(0.0);
    assertThat(getProjectMeasure("file_complexity_distribution").getData()).isEqualTo("0=1;5=0;10=0;20=0;30=0;60=0;90=0");
    // Duplication
    assertThat(getProjectMeasure("duplicated_lines").getValue()).isEqualTo(0.0);
    assertThat(getProjectMeasure("duplicated_blocks").getValue()).isEqualTo(0.0);
    assertThat(getProjectMeasure("duplicated_files").getValue()).isEqualTo(0.0);
    assertThat(getProjectMeasure("duplicated_lines_density").getValue()).isEqualTo(0.0);
    // Rules
    assertThat(getProjectMeasure("violations").getValue()).isEqualTo(0.0);

    assertThat(getProjectMeasure("tests")).isNull();
    assertThat(getProjectMeasure("coverage")).isNull();
  }

  @Test
  public void directory_level() {
    // Size
    assertThat(getDirectoryMeasure("ncloc").getIntValue()).isEqualTo(1);
    assertThat(getDirectoryMeasure("lines").getIntValue()).isEqualTo(6);
    assertThat(getDirectoryMeasure("files").getIntValue()).isEqualTo(1);
    assertThat(getDirectoryMeasure("directories").getIntValue()).isEqualTo(1);
    assertThat(getDirectoryMeasure("statements").getIntValue()).isEqualTo(1);
    assertThat(getDirectoryMeasure("functions").getIntValue()).isEqualTo(0);
    assertThat(getDirectoryMeasure("classes").getIntValue()).isEqualTo(0);
    // Documentation
    assertThat(getDirectoryMeasure("comment_lines").getIntValue()).isEqualTo(1);
    assertThat(getDirectoryMeasure("comment_lines_density").getValue()).isEqualTo(50.0);
    // Complexity
    assertThat(getDirectoryMeasure("complexity").getValue()).isEqualTo(0.0);
    assertThat(getDirectoryMeasure("function_complexity")).isNull();
    assertThat(getProjectMeasure("function_complexity_distribution").getData()).isEqualTo("1=0;2=0;4=0;6=0;8=0;10=0;12=0;20=0;30=0");
    assertThat(getDirectoryMeasure("file_complexity").getValue()).isEqualTo(0.0);
    assertThat(getDirectoryMeasure("file_complexity_distribution").getData()).isEqualTo("0=1;5=0;10=0;20=0;30=0;60=0;90=0");
    // Duplication
    assertThat(getDirectoryMeasure("duplicated_lines").getValue()).isEqualTo(0.0);
    assertThat(getDirectoryMeasure("duplicated_blocks").getValue()).isEqualTo(0.0);
    assertThat(getDirectoryMeasure("duplicated_files").getValue()).isEqualTo(0.0);
    assertThat(getDirectoryMeasure("duplicated_lines_density").getValue()).isEqualTo(0.0);
    // Rules
    assertThat(getDirectoryMeasure("violations").getValue()).isEqualTo(0.0);
  }

  @Test
  public void file_level() {
    // Size
    assertThat(getFileMeasure("ncloc").getIntValue()).isEqualTo(1);
    assertThat(getFileMeasure("lines").getIntValue()).isEqualTo(6);
    assertThat(getFileMeasure("files").getIntValue()).isEqualTo(1);
    assertThat(getFileMeasure("statements").getIntValue()).isEqualTo(1);
    assertThat(getFileMeasure("functions").getIntValue()).isEqualTo(0);
    assertThat(getFileMeasure("classes").getIntValue()).isEqualTo(0);
    // Documentation
    assertThat(getFileMeasure("comment_lines").getIntValue()).isEqualTo(1);
    assertThat(getFileMeasure("comment_lines_density").getValue()).isEqualTo(50.0);
    // Complexity
    assertThat(getFileMeasure("complexity").getValue()).isEqualTo(0.0);
    assertThat(getFileMeasure("function_complexity")).isNull();
    assertThat(getFileMeasure("function_complexity_distribution")).isNull();
    assertThat(getFileMeasure("file_complexity").getValue()).isEqualTo(0.0);
    assertThat(getFileMeasure("file_complexity_distribution")).isNull();
    // Duplication
    assertThat(getFileMeasure("duplicated_lines")).isNull();
    assertThat(getFileMeasure("duplicated_blocks")).isNull();
    assertThat(getFileMeasure("duplicated_files")).isNull();
    assertThat(getFileMeasure("duplicated_lines_density")).isNull();
    // Rules
    assertThat(getFileMeasure("violations")).isNull();
  }

  /**
   * SONARPLUGINS-2184
   */
  @Test
  public void should_be_compatible_with_DevCockpit() {
    // TODO probably bug in Sonar: order might depend on JVM
    assertThat(getFileMeasure("ncloc_data").getData())
      .contains("1=0")
      .contains("5=1");
    assertThat(getFileMeasure("comment_lines_data").getData())
      .contains("1=1")
      .contains("4=0");
  }

  /* Helper methods */

  private Measure getProjectMeasure(String metricKey) {
    Resource resource = wsClient.find(ResourceQuery.createForMetrics(PROJECT_KEY, metricKey));
    return resource == null ? null : resource.getMeasure(metricKey);
  }

  private Measure getDirectoryMeasure(String metricKey) {
    Resource resource = wsClient.find(ResourceQuery.createForMetrics(keyFor("dir"), metricKey));
    return resource == null ? null : resource.getMeasure(metricKey);
  }

  private Measure getFileMeasure(String metricKey) {
    Resource resource = wsClient.find(ResourceQuery.createForMetrics(keyFor("dir/HelloWorld.py"), metricKey));
    return resource == null ? null : resource.getMeasure(metricKey);
  }

  private static String keyFor(String s) {
    return PROJECT_KEY + ":src/" + s;
  }

}
