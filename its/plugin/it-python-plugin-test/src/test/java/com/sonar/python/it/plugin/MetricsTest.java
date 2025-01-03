/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2025 SonarSource SA
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

import com.sonar.orchestrator.build.BuildResult;
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import static com.sonar.python.it.plugin.TestsUtils.getMeasureAsDouble;
import static com.sonar.python.it.plugin.TestsUtils.getMeasureAsInt;
import static org.assertj.core.api.Assertions.assertThat;

class MetricsTest {

  private static final String PROJECT_KEY = "metrics";

  private static final String CLASSES = "classes";
  private static final String COGNITIVE_COMPLEXITY = "cognitive_complexity";
  private static final String COMMENT_LINES = "comment_lines";
  private static final String COMMENT_LINES_DENSITY = "comment_lines_density";
  private static final String COMPLEXITY = "complexity";
  private static final String COVERAGE = "coverage";
  private static final String DUPLICATED_BLOCKS = "duplicated_blocks";
  private static final String DUPLICATED_FILES = "duplicated_files";
  private static final String DUPLICATED_LINES = "duplicated_lines";
  private static final String DUPLICATED_LINES_DENSITY = "duplicated_lines_density";
  private static final String FILES = "files";
  private static final String FUNCTIONS = "functions";
  private static final String LINES = "lines";
  private static final String NCLOC = "ncloc";
  private static final String STATEMENTS = "statements";
  private static final String TESTS = "tests";
  private static final String VIOLATIONS = "violations";

  private static final String HELLO_WORLD_PY = "dir/HelloWorld.py";

  private static final Offset<Double> OFFSET = Offset.offset(0.01d);

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension orchestrator = TestsUtils.ORCHESTRATOR;

  private static BuildResult buildResult;

  @BeforeAll
  static void startServer() {
    orchestrator.getServer().provisionProject(PROJECT_KEY, PROJECT_KEY);
    orchestrator.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "no_rule");
    SonarScanner build = orchestrator.createSonarScanner()
      .setProjectDir(new File("projects/metrics"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1.0-SNAPSHOT")
      .setTestDirs("test")
      .setSourceDirs("src");
    buildResult = orchestrator.executeBuild(build);
  }

  @Test
  void project_level() {
    // Size
    assertThat(getProjectMeasureAsInt(NCLOC)).isEqualTo(6);
    assertThat(getProjectMeasureAsInt(LINES)).isEqualTo(13);
    assertThat(getProjectMeasureAsInt(FILES)).isEqualTo(2);
    assertThat(getProjectMeasureAsInt(STATEMENTS)).isEqualTo(6);
    assertThat(getProjectMeasureAsInt(FUNCTIONS)).isEqualTo(1);
    assertThat(getProjectMeasureAsInt(CLASSES)).isEqualTo(0);
    // Documentation
    assertThat(getProjectMeasureAsInt(COMMENT_LINES)).isEqualTo(1);
    assertThat(getProjectMeasureAsDouble(COMMENT_LINES_DENSITY)).isEqualTo(14.3, OFFSET);
    // Complexity
    assertThat(getProjectMeasureAsDouble(COMPLEXITY)).isEqualTo(3.0, OFFSET);
    assertThat(getProjectMeasureAsDouble(COGNITIVE_COMPLEXITY)).isEqualTo(3.0, OFFSET);
    // Duplication
    assertThat(getProjectMeasureAsDouble(DUPLICATED_LINES)).isEqualTo(0.0, OFFSET);
    assertThat(getProjectMeasureAsDouble(DUPLICATED_BLOCKS)).isEqualTo(0.0, OFFSET);
    assertThat(getProjectMeasureAsDouble(DUPLICATED_FILES)).isEqualTo(0.0, OFFSET);
    assertThat(getProjectMeasureAsDouble(DUPLICATED_LINES_DENSITY)).isEqualTo(0.0, OFFSET);
    // Rules
    assertThat(getProjectMeasureAsDouble(VIOLATIONS)).isEqualTo(0.0, OFFSET);

    assertThat(getProjectMeasureAsInt(TESTS)).isNull();
    assertThat(getProjectMeasureAsDouble(COVERAGE)).isEqualTo(0.0, OFFSET);
  }

  @Test
  void directory_level() {
    // Size
    assertThat(getDirectoryMeasureAsInt(NCLOC)).isEqualTo(6);
    assertThat(getDirectoryMeasureAsInt(LINES)).isEqualTo(13);
    assertThat(getDirectoryMeasureAsInt(FILES)).isEqualTo(2);
    assertThat(getDirectoryMeasureAsInt(STATEMENTS)).isEqualTo(6);
    assertThat(getDirectoryMeasureAsInt(FUNCTIONS)).isEqualTo(1);
    assertThat(getDirectoryMeasureAsInt(CLASSES)).isEqualTo(0);
    // Documentation
    assertThat(getDirectoryMeasureAsInt(COMMENT_LINES)).isEqualTo(1);
    assertThat(getDirectoryMeasureAsDouble(COMMENT_LINES_DENSITY)).isEqualTo(14.3, OFFSET);
    // Complexity
    assertThat(getDirectoryMeasureAsDouble(COMPLEXITY)).isEqualTo(3.0, OFFSET);
    assertThat(getDirectoryMeasureAsDouble(COGNITIVE_COMPLEXITY)).isEqualTo(3.0, OFFSET);
    // Duplication
    assertThat(getDirectoryMeasureAsDouble(DUPLICATED_LINES)).isEqualTo(0.0, OFFSET);
    assertThat(getDirectoryMeasureAsDouble(DUPLICATED_BLOCKS)).isEqualTo(0.0, OFFSET);
    assertThat(getDirectoryMeasureAsDouble(DUPLICATED_FILES)).isEqualTo(0.0, OFFSET);
    assertThat(getDirectoryMeasureAsDouble(DUPLICATED_LINES_DENSITY)).isEqualTo(0.0, OFFSET);
    // Rules
    assertThat(getDirectoryMeasureAsDouble(VIOLATIONS)).isEqualTo(0.0, OFFSET);
  }

  @Test
  void file_level() {
    // Size
    assertThat(getFileMeasureAsInt(NCLOC)).isEqualTo(1);
    assertThat(getFileMeasureAsInt(LINES)).isEqualTo(6);
    assertThat(getFileMeasureAsInt(FILES)).isEqualTo(1);
    assertThat(getFileMeasureAsInt(STATEMENTS)).isEqualTo(2);
    assertThat(getFileMeasureAsInt(FUNCTIONS)).isEqualTo(1);
    assertThat(getFileMeasureAsInt(CLASSES)).isEqualTo(0);
    // Documentation
    assertThat(getFileMeasureAsInt(COMMENT_LINES)).isEqualTo(1);
    assertThat(getFileMeasureAsDouble(COMMENT_LINES_DENSITY)).isEqualTo(50.0, OFFSET);
    // Complexity
    assertThat(getFileMeasureAsDouble(COMPLEXITY)).isEqualTo(2.0, OFFSET);
    assertThat(getFileMeasureAsDouble(COGNITIVE_COMPLEXITY)).isEqualTo(1.0, OFFSET);
    // Duplication
    assertThat(getFileMeasureAsInt(DUPLICATED_LINES)).isZero();
    assertThat(getFileMeasureAsInt(DUPLICATED_BLOCKS)).isZero();
    assertThat(getFileMeasureAsInt(DUPLICATED_FILES)).isZero();
    assertThat(getFileMeasureAsDouble(DUPLICATED_LINES_DENSITY)).isZero();
    // Rules
    assertThat(getFileMeasureAsInt(VIOLATIONS)).isZero();
  }

  /* Helper methods */

  private Integer getProjectMeasureAsInt(String metricKey) {
    return getMeasureAsInt(PROJECT_KEY, metricKey);
  }

  private Double getProjectMeasureAsDouble(String metricKey) {
    return getMeasureAsDouble(PROJECT_KEY, metricKey);
  }

  private Integer getDirectoryMeasureAsInt(String metricKey) {
    return getMeasureAsInt(keyFor("dir"), metricKey);
  }

  private Double getDirectoryMeasureAsDouble(String metricKey) {
    return getMeasureAsDouble(keyFor("dir"), metricKey);
  }

  private Integer getFileMeasureAsInt(String metricKey) {
    return getMeasureAsInt(keyFor(HELLO_WORLD_PY), metricKey);
  }

  private Double getFileMeasureAsDouble(String metricKey) {
    return getMeasureAsDouble(keyFor(HELLO_WORLD_PY), metricKey);
  }

  private static String keyFor(String s) {
    return PROJECT_KEY + ":src/" + s;
  }

}
