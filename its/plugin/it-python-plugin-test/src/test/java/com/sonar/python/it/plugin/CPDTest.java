/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2023 SonarSource SA
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

import com.sonar.orchestrator.build.SonarScanner;
import com.sonar.orchestrator.junit5.OrchestratorExtension;
import java.io.File;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

import static com.sonar.python.it.plugin.Tests.getMeasureAsDouble;
import static org.assertj.core.api.Assertions.assertThat;

public class CPDTest {

  private static final String PROJECT_KEY = "cpd";

  private static final String FILE_1_KEY = PROJECT_KEY + ":file1.py";
  private static final String FILE_2_KEY = PROJECT_KEY + ":file2.py";
  private static final String FILE_3_KEY = PROJECT_KEY + ":file3.py";

  private static final String DUPLICATED_LINES = "duplicated_lines";
  private static final String DUPLICATED_BLOCKS = "duplicated_blocks";
  private static final String DUPLICATED_FILES = "duplicated_files";
  private static final String DUPLICATED_LINES_DENSITY = "duplicated_lines_density";

  @RegisterExtension
  public static final OrchestratorExtension ORCHESTRATOR = Tests.ORCHESTRATOR;

  @BeforeAll
  public static void startServer() {
    ORCHESTRATOR.getServer().provisionProject(PROJECT_KEY, PROJECT_KEY);
    ORCHESTRATOR.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "no_rule");
    SonarScanner build = SonarScanner.create()
      .setProjectDir(new File("projects/cpd"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1.0-SNAPSHOT")
      .setSourceDirs(".")
      .setSourceEncoding("UTF-8");
    ORCHESTRATOR.executeBuild(build);
  }

  @Test
  public void test_cpd() {
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
