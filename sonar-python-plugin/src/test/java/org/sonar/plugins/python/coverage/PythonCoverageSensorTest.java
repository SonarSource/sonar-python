/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.plugins.python.coverage;

import com.google.common.collect.ImmutableSet;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.FileMetadata;
import org.sonar.api.batch.sensor.coverage.CoverageType;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.config.Settings;
import org.sonar.api.internal.google.common.base.Charsets;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonCoverageSensorTest {


  private final String FILE1_KEY = "moduleKey:sources/file1.py";
  private final String FILE2_KEY = "moduleKey:sources/file2.py";
  private final String FILE3_KEY = "moduleKey:sources/file3.py";
  private SensorContextTester context;
  private Settings settings;
  private Map<InputFile, Set<Integer>> linesOfCode;

  private PythonCoverageSensor coverageSensor = new PythonCoverageSensor();
  private File moduleBaseDir = new File("src/test/resources/org/sonar/plugins/python/coverage-reports");

  @Before
  public void init() {
    settings = new Settings();
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage.xml");
    settings.setProperty(PythonCoverageSensor.IT_REPORT_PATH_KEY, "coverage.xml");
    settings.setProperty(PythonCoverageSensor.OVERALL_REPORT_PATH_KEY, "coverage.xml");
    context = SensorContextTester.create(moduleBaseDir);
    context.setSettings(settings);

    InputFile inputFile1 = inputFile("sources/file1.py", Type.MAIN);
    InputFile inputFile2 = inputFile("sources/file2.py", Type.MAIN);
    InputFile inputFile3 = inputFile("sources/file3.py", Type.MAIN);

    linesOfCode = new HashMap<>();
    linesOfCode.put(inputFile1, ImmutableSet.of(1, 4, 6));
    linesOfCode.put(inputFile2, ImmutableSet.of(1, 2, 3, 4, 5, 6));
    linesOfCode.put(inputFile3, ImmutableSet.of(1, 3));
  }

  private InputFile inputFile(String relativePath, Type type) {
    DefaultInputFile inputFile = new DefaultInputFile("moduleKey", relativePath)
      .setModuleBaseDir(moduleBaseDir.toPath())
      .setLanguage("py")
      .setType(type);

    inputFile.initMetadata(new FileMetadata().readMetadata(inputFile.file(), Charsets.UTF_8));
    context.fileSystem().add(inputFile);

    return inputFile;
  }

  @Test
  public void report_not_found() throws Exception {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "/fake/path/report.xml");

    coverageSensor.execute(context, linesOfCode);

    // expected logged text: "No report was found for sonar.python.coverage.reportPath using pattern /fake/path/report.xml"
    assertThat(context.lineHits(FILE1_KEY, CoverageType.UNIT, 1)).isNull();
  }

  @Test
  public void absolute_path() throws Exception {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, new File(moduleBaseDir, "coverage.xml").getAbsolutePath());

    coverageSensor.execute(context, linesOfCode);

    assertThat(context.lineHits(FILE1_KEY, CoverageType.UNIT, 1)).isEqualTo(1);
  }

  @Test
  public void test_coverage() {
    coverageSensor.execute(context, linesOfCode);
    Integer[] file1Expected = {1, null, null, 0, null, 0};
    Integer[] file2Expected = {1, 3, 1, 0, 1, 1};

    for (int line = 1; line <= 6; line++) {
      assertThat(context.lineHits(FILE1_KEY, CoverageType.UNIT, line)).isEqualTo(file1Expected[line - 1]);
      assertThat(context.lineHits(FILE1_KEY, CoverageType.IT, line)).isEqualTo(file1Expected[line - 1]);
      assertThat(context.lineHits(FILE1_KEY, CoverageType.OVERALL, line)).isEqualTo(file1Expected[line - 1]);

      assertThat(context.lineHits(FILE2_KEY, CoverageType.UNIT, line)).isEqualTo(file2Expected[line - 1]);
      assertThat(context.lineHits(FILE2_KEY, CoverageType.IT, line)).isEqualTo(file2Expected[line - 1]);
      assertThat(context.lineHits(FILE2_KEY, CoverageType.OVERALL, line)).isEqualTo(file2Expected[line - 1]);

      assertThat(context.lineHits(FILE3_KEY, CoverageType.UNIT, line)).isNull();
      assertThat(context.lineHits(FILE3_KEY, CoverageType.IT, line)).isNull();
      assertThat(context.lineHits(FILE3_KEY, CoverageType.OVERALL, line)).isNull();
    }

    assertThat(context.conditions(FILE2_KEY, CoverageType.UNIT, 2)).isNull();
    assertThat(context.conditions(FILE2_KEY, CoverageType.UNIT, 3)).isEqualTo(2);
    assertThat(context.coveredConditions(FILE2_KEY, CoverageType.UNIT, 3)).isEqualTo(1);
  }
  @Test
  public void test_unresolved_path() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage_with_unresolved_path.xml");
    coverageSensor.execute(context, linesOfCode);

    assertThat(context.lineHits(FILE1_KEY, CoverageType.UNIT, 1)).isEqualTo(1);
  }

  @Test
  public void test_force_zero_coverage_no_report() {
    Settings newSettings = new Settings().setProperty(PythonCoverageSensor.FORCE_ZERO_COVERAGE_KEY, "true");
    context.setSettings(newSettings);
    coverageSensor.execute(context, linesOfCode);
    assertThat(context.lineHits(FILE1_KEY, CoverageType.UNIT, 1)).isEqualTo(0);
    assertThat(context.lineHits(FILE3_KEY, CoverageType.UNIT, 1)).isEqualTo(0);

    context.setSettings(newSettings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage.xml"));
    coverageSensor.execute(context, linesOfCode);
    assertThat(context.lineHits(FILE1_KEY, CoverageType.UNIT, 1)).isEqualTo(1);
    assertThat(context.lineHits(FILE3_KEY, CoverageType.UNIT, 1)).isEqualTo(0);
  }

  @Test
  public void test_force_zero_coverage_no_lines_of_code() throws Exception {
    Settings newSettings = new Settings().setProperty(PythonCoverageSensor.FORCE_ZERO_COVERAGE_KEY, "true");
    context.setSettings(newSettings);
    coverageSensor.execute(context, new HashMap<>());
    assertThat(context.lineHits(FILE1_KEY, CoverageType.UNIT, 1)).isNull();
  }

  @Test(expected = IllegalStateException.class)
  public void should_fail_on_invalid_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "invalid-coverage-result.xml");
    coverageSensor.execute(context, linesOfCode);
  }

  @Test(expected = IllegalStateException.class)
  public void should_fail_on_unexpected_eof() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage_with_eof_error.xml");
    coverageSensor.execute(context, linesOfCode);
  }

  @Test
  public void should_do_nothing_on_empty_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "empty-coverage-result.xml");
    settings.setProperty(PythonCoverageSensor.IT_REPORT_PATH_KEY, "this-file-does-not-exist.xml");
    coverageSensor.execute(context, linesOfCode);

    assertThat(context.lineHits(FILE1_KEY, CoverageType.UNIT, 1)).isNull();
  }
}
