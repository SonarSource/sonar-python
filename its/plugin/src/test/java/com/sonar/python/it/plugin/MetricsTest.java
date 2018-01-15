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
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.sonarqube.ws.WsMeasures.Measure;

import static com.sonar.python.it.plugin.Tests.getMeasure;
import static com.sonar.python.it.plugin.Tests.getMeasureAsDouble;
import static com.sonar.python.it.plugin.Tests.getMeasureAsInt;
import static org.assertj.core.api.Assertions.assertThat;

public class MetricsTest {

  private static final String PROJECT_KEY = "metrics";

  @ClassRule
  public static Orchestrator orchestrator = Tests.ORCHESTRATOR;

  @BeforeClass
  public static void startServer() {
    orchestrator.getServer().provisionProject(PROJECT_KEY, PROJECT_KEY);
    orchestrator.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "no_rule");
    SonarScanner build = SonarScanner.create()
      .setProjectDir(new File("projects/metrics"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1.0-SNAPSHOT")
      .setSourceDirs("src");
    orchestrator.executeBuild(build);
  }

  @Test
  public void project_level() {
    // Size
    assertThat(getProjectMeasureAsInt("ncloc")).isEqualTo(1);
    assertThat(getProjectMeasureAsInt("lines")).isEqualTo(6);
    assertThat(getProjectMeasureAsInt("files")).isEqualTo(1);
    assertThat(getProjectMeasureAsInt("statements")).isEqualTo(1);
    assertThat(getProjectMeasureAsInt("directories")).isEqualTo(1);
    assertThat(getProjectMeasureAsInt("functions")).isEqualTo(0);
    assertThat(getProjectMeasureAsInt("classes")).isEqualTo(0);
    // Documentation
    assertThat(getProjectMeasureAsInt("comment_lines")).isEqualTo(1);
    assertThat(getProjectMeasureAsDouble("comment_lines_density")).isEqualTo(50.0);
    // Complexity
    assertThat(getProjectMeasureAsDouble("complexity")).isZero();
    assertThat(getProjectMeasureAsDouble("function_complexity")).isNull();
    assertThat(getProjectMeasure("function_complexity_distribution").getValue()).isEqualTo("1=0;2=0;4=0;6=0;8=0;10=0;12=0;20=0;30=0");
    assertThat(getProjectMeasureAsDouble("file_complexity")).isZero();
    assertThat(getProjectMeasure("file_complexity_distribution").getValue()).isEqualTo("0=1;5=0;10=0;20=0;30=0;60=0;90=0");
    // Duplication
    assertThat(getProjectMeasureAsDouble("duplicated_lines")).isZero();
    assertThat(getProjectMeasureAsDouble("duplicated_blocks")).isZero();
    assertThat(getProjectMeasureAsDouble("duplicated_files")).isZero();
    assertThat(getProjectMeasureAsDouble("duplicated_lines_density")).isZero();
    // Rules
    assertThat(getProjectMeasureAsDouble("violations")).isZero();

    assertThat(getProjectMeasureAsInt("tests")).isNull();
    assertThat(getProjectMeasureAsDouble("coverage")).isZero();
  }

  @Test
  public void directory_level() {
    // Size
    assertThat(getDirectoryMeasureAsInt("ncloc")).isEqualTo(1);
    assertThat(getDirectoryMeasureAsInt("lines")).isEqualTo(6);
    assertThat(getDirectoryMeasureAsInt("files")).isEqualTo(1);
    assertThat(getDirectoryMeasureAsInt("directories")).isEqualTo(1);
    assertThat(getDirectoryMeasureAsInt("statements")).isEqualTo(1);
    assertThat(getDirectoryMeasureAsInt("functions")).isEqualTo(0);
    assertThat(getDirectoryMeasureAsInt("classes")).isEqualTo(0);
    // Documentation
    assertThat(getDirectoryMeasureAsInt("comment_lines")).isEqualTo(1);
    assertThat(getDirectoryMeasureAsDouble("comment_lines_density")).isEqualTo(50.0);
    // Complexity
    assertThat(getDirectoryMeasureAsDouble("complexity")).isZero();
    assertThat(getDirectoryMeasureAsDouble("function_complexity")).isNull();
    assertThat(getDirectoryMeasure("function_complexity_distribution").getValue()).isEqualTo("1=0;2=0;4=0;6=0;8=0;10=0;12=0;20=0;30=0");
    assertThat(getDirectoryMeasureAsDouble("file_complexity")).isZero();
    assertThat(getDirectoryMeasure("file_complexity_distribution").getValue()).isEqualTo("0=1;5=0;10=0;20=0;30=0;60=0;90=0");
    // Duplication
    assertThat(getDirectoryMeasureAsDouble("duplicated_lines")).isZero();
    assertThat(getDirectoryMeasureAsDouble("duplicated_blocks")).isZero();
    assertThat(getDirectoryMeasureAsDouble("duplicated_files")).isZero();
    assertThat(getDirectoryMeasureAsDouble("duplicated_lines_density")).isZero();
    // Rules
    assertThat(getDirectoryMeasureAsDouble("violations")).isZero();
  }

  @Test
  public void file_level() {
    // Size
    assertThat(getFileMeasureAsInt("ncloc")).isEqualTo(1);
    assertThat(getFileMeasureAsInt("lines")).isEqualTo(6);
    assertThat(getFileMeasureAsInt("files")).isEqualTo(1);
    assertThat(getFileMeasureAsInt("statements")).isEqualTo(1);
    assertThat(getFileMeasureAsInt("functions")).isEqualTo(0);
    assertThat(getFileMeasureAsInt("classes")).isEqualTo(0);
    // Documentation
    assertThat(getFileMeasureAsInt("comment_lines")).isEqualTo(1);
    assertThat(getFileMeasureAsDouble("comment_lines_density")).isEqualTo(50.0);
    // Complexity
    assertThat(getFileMeasureAsDouble("complexity")).isZero();
    assertThat(getFileMeasureAsDouble("function_complexity")).isNull();
    assertThat(getFileMeasureAsDouble("function_complexity_distribution")).isNull();
    assertThat(getFileMeasureAsDouble("file_complexity")).isZero();
    assertThat(getFileMeasureAsDouble("file_complexity_distribution")).isNull();
    // Duplication
    assertThat(getFileMeasureAsInt("duplicated_lines")).isZero();
    assertThat(getFileMeasureAsInt("duplicated_blocks")).isZero();
    assertThat(getFileMeasureAsInt("duplicated_files")).isZero();
    assertThat(getFileMeasureAsDouble("duplicated_lines_density")).isZero();
    // Rules
    assertThat(getFileMeasureAsInt("violations")).isZero();
  }

  /**
   * SONARPLUGINS-2184
   */
  @Test
  public void should_be_compatible_with_DevCockpit() {
    // TODO probably bug in Sonar: order might depend on JVM
    assertThat(getFileMeasure("ncloc_data").getValue())
      .doesNotContain("1=1")
      .contains("5=1");
    assertThat(getFileMeasure("comment_lines_data").getValue())
      .contains("2=1")
      .doesNotContain("4=1");
    assertThat(getFileMeasure("executable_lines_data").getValue())
      .doesNotContain("1=1")
      .contains("5=1");
  }

  /* Helper methods */

  private Measure getProjectMeasure(String metricKey) {
    return getMeasure(PROJECT_KEY, metricKey);
  }

  private Integer getProjectMeasureAsInt(String metricKey) {
    return getMeasureAsInt(PROJECT_KEY, metricKey);
  }

  private Double getProjectMeasureAsDouble(String metricKey) {
    return getMeasureAsDouble(PROJECT_KEY, metricKey);
  }

  private Measure getDirectoryMeasure(String metricKey) {
    return getMeasure(keyFor("dir"), metricKey);
  }

  private Integer getDirectoryMeasureAsInt(String metricKey) {
    return getMeasureAsInt(keyFor("dir"), metricKey);
  }

  private Double getDirectoryMeasureAsDouble(String metricKey) {
    return getMeasureAsDouble(keyFor("dir"), metricKey);
  }

  private Measure getFileMeasure(String metricKey) {
    return getMeasure(keyFor("dir/HelloWorld.py"), metricKey);
  }

  private Integer getFileMeasureAsInt(String metricKey) {
    return getMeasureAsInt(keyFor("dir/HelloWorld.py"), metricKey);
  }

  private Double getFileMeasureAsDouble(String metricKey) {
    return getMeasureAsDouble(keyFor("dir/HelloWorld.py"), metricKey);
  }

  private static String keyFor(String s) {
    return PROJECT_KEY + ":src/" + s;
  }

}
