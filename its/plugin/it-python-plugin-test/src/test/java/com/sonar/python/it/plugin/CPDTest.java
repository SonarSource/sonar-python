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

import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import static com.sonar.python.it.plugin.TestsUtils.getMeasureAsDouble;
import static org.assertj.core.api.Assertions.assertThat;

class CPDTest {

  private static final String PROJECT_KEY = "cpd";

  private static final String FILE_1_KEY = PROJECT_KEY + ":file1.py";
  private static final String FILE_2_KEY = PROJECT_KEY + ":file2.py";
  private static final String FILE_3_KEY = PROJECT_KEY + ":file3.py";

  private static final String DUPLICATED_LINES = "duplicated_lines";
  private static final String DUPLICATED_BLOCKS = "duplicated_blocks";
  private static final String DUPLICATED_FILES = "duplicated_files";
  private static final String DUPLICATED_LINES_DENSITY = "duplicated_lines_density";

  @RegisterExtension
  public static final ConcurrentOrchestratorExtension ORCHESTRATOR = TestsUtils.ORCHESTRATOR;

  @BeforeAll
  static void startServer() {
    ORCHESTRATOR.getServer().provisionProject(PROJECT_KEY, PROJECT_KEY);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "no_rule");
    SonarScanner build = ORCHESTRATOR.createSonarScanner()
      .setProjectDir(new File("projects/cpd"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1.0-SNAPSHOT")
      .setSourceDirs(".")
      .setSourceEncoding("UTF-8");
    ORCHESTRATOR.executeBuild(build);
  }

  @Test
  void test_cpd() {
    Offset<Double> offset = Offset.offset(0.01d);

    assertThat(getMeasureAsDouble(FILE_1_KEY, DUPLICATED_LINES)).isEqualTo(27.0d, offset);
    assertThat(getMeasureAsDouble(FILE_1_KEY, DUPLICATED_BLOCKS)).isEqualTo(1.0d, offset);
    assertThat(getMeasureAsDouble(FILE_1_KEY, DUPLICATED_FILES)).isEqualTo(1.0d, offset);
    assertThat(getMeasureAsDouble(FILE_1_KEY, DUPLICATED_LINES_DENSITY)).isEqualTo(93.1d, offset);

    assertThat(getMeasureAsDouble(FILE_2_KEY, DUPLICATED_LINES)).isEqualTo(27.0d, offset);
    assertThat(getMeasureAsDouble(FILE_2_KEY, DUPLICATED_BLOCKS)).isEqualTo(1.0d, offset);
    assertThat(getMeasureAsDouble(FILE_2_KEY, DUPLICATED_FILES)).isEqualTo(1.0d, offset);
    assertThat(getMeasureAsDouble(FILE_2_KEY, DUPLICATED_LINES_DENSITY)).isEqualTo(90.0d, offset);

    assertThat(getMeasureAsDouble(FILE_3_KEY, DUPLICATED_LINES)).isEqualTo(0.0d, offset);
    assertThat(getMeasureAsDouble(FILE_3_KEY, DUPLICATED_BLOCKS)).isEqualTo(0.0d, offset);
    assertThat(getMeasureAsDouble(FILE_3_KEY, DUPLICATED_FILES)).isEqualTo(0.0d, offset);
    assertThat(getMeasureAsDouble(FILE_3_KEY, DUPLICATED_LINES_DENSITY)).isEqualTo(0.0d, offset);

    assertThat(getMeasureAsDouble(PROJECT_KEY, DUPLICATED_LINES)).isEqualTo(54.0d, offset);
    assertThat(getMeasureAsDouble(PROJECT_KEY, DUPLICATED_BLOCKS)).isEqualTo(2.0d, offset);
    assertThat(getMeasureAsDouble(PROJECT_KEY, DUPLICATED_FILES)).isEqualTo(2.0d, offset);
    assertThat(getMeasureAsDouble(PROJECT_KEY, DUPLICATED_LINES_DENSITY)).isEqualTo(61.4d, offset);
 }

}
