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

import com.sonar.orchestrator.Orchestrator;
import com.sonar.orchestrator.build.SonarScanner;
import java.io.File;
import org.assertj.core.data.Offset;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;

import static com.sonar.python.it.plugin.Tests.getMeasureAsDouble;
import static org.assertj.core.api.Assertions.assertThat;

public class CPDTest {

  private static final String PROJECT_KEY = "cpd";

  @ClassRule
  public static Orchestrator orchestrator = Tests.ORCHESTRATOR;

  @BeforeClass
  public static void startServer() {
    orchestrator.getServer().provisionProject(PROJECT_KEY, PROJECT_KEY);
    orchestrator.getServer().associateProjectToQualityProfile(PROJECT_KEY, "py", "no_rule");
    SonarScanner build = SonarScanner.create()
      .setProjectDir(new File("projects/cpd"))
      .setProjectKey(PROJECT_KEY)
      .setProjectName(PROJECT_KEY)
      .setProjectVersion("1.0-SNAPSHOT")
      .setSourceDirs(".")
      .setSourceEncoding("UTF-8");
    orchestrator.executeBuild(build);
  }

  @Test
  public void test_cpd() {
    Offset<Double> offset = Offset.offset(0.01d);

    assertThat(getFileMeasureAsDouble("file1.py","duplicated_lines")).isEqualTo(27.0d, offset);
    assertThat(getFileMeasureAsDouble("file1.py","duplicated_blocks")).isEqualTo(1.0d, offset);
    assertThat(getFileMeasureAsDouble("file1.py","duplicated_files")).isEqualTo(1.0d, offset);
    assertThat(getFileMeasureAsDouble("file1.py","duplicated_lines_density")).isEqualTo(96.4d, offset);

    assertThat(getFileMeasureAsDouble("file2.py","duplicated_lines")).isEqualTo(27.0d, offset);
    assertThat(getFileMeasureAsDouble("file2.py","duplicated_blocks")).isEqualTo(1.0d, offset);
    assertThat(getFileMeasureAsDouble("file2.py","duplicated_files")).isEqualTo(1.0d, offset);
    assertThat(getFileMeasureAsDouble("file2.py","duplicated_lines_density")).isEqualTo(96.4d, offset);

    assertThat(getFileMeasureAsDouble("file3.py","duplicated_lines")).isEqualTo(0.0d, offset);
    assertThat(getFileMeasureAsDouble("file3.py","duplicated_blocks")).isEqualTo(0.0d, offset);
    assertThat(getFileMeasureAsDouble("file3.py","duplicated_files")).isEqualTo(0.0d, offset);
    assertThat(getFileMeasureAsDouble("file3.py","duplicated_lines_density")).isEqualTo(0.0d, offset);

    assertThat(getMeasureAsDouble(PROJECT_KEY, "duplicated_lines")).isEqualTo(54.0d, offset);
    assertThat(getMeasureAsDouble(PROJECT_KEY, "duplicated_blocks")).isEqualTo(2.0d, offset);
    assertThat(getMeasureAsDouble(PROJECT_KEY, "duplicated_files")).isEqualTo(2.0d, offset);
    assertThat(getMeasureAsDouble(PROJECT_KEY, "duplicated_lines_density")).isEqualTo(63.5d, offset);
 }

  private Double getFileMeasureAsDouble(String path, String metricKey) {
    return getMeasureAsDouble(PROJECT_KEY + ":" + path, metricKey);
  }

}
