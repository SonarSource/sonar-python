/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.plugins.python.coverage;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.config.Settings;
import org.sonar.api.config.internal.MapSettings;
import org.sonar.plugins.python.TestUtils;

import static org.assertj.core.api.Assertions.assertThat;

;

public class PythonCoverageSensorTest {

  private final String FILE1_KEY = "moduleKey:sources/file1.py";
  private final String FILE2_KEY = "moduleKey:sources/file2.py";
  private final String FILE3_KEY = "moduleKey:sources/file3.py";
  private final String FILE4_KEY = "moduleKey:sources/file4.py";
  private SensorContextTester context;
  private Settings settings;

  private PythonCoverageSensor coverageSensor = new PythonCoverageSensor();
  private File moduleBaseDir = new File("src/test/resources/org/sonar/plugins/python/coverage-reports").getAbsoluteFile();

  @Before
  public void init() {
    settings = new MapSettings();
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage.xml");
    context = SensorContextTester.create(moduleBaseDir);
    context.setSettings(settings);

    inputFile("sources/file1.py", Type.MAIN);
    inputFile("sources/file2.py", Type.MAIN);
    inputFile("sources/file3.py", Type.MAIN);
    inputFile("sources/file4.py", Type.MAIN);
    inputFile("sources/folder1/file1.py", Type.MAIN);
    inputFile("sources/folder1/file2.py", Type.MAIN);
    inputFile("sources/folder2/file2.py", Type.MAIN);
  }

  private InputFile inputFile(String relativePath, Type type) {
    DefaultInputFile inputFile = TestInputFileBuilder.create("moduleKey", relativePath)
      .setModuleBaseDir(moduleBaseDir.toPath())
      .setLanguage("py")
      .setType(type)
      .initMetadata(TestUtils.fileContent(new File(moduleBaseDir, relativePath), StandardCharsets.UTF_8))
      .build();

    context.fileSystem().add(inputFile);

    return inputFile;
  }

  @Test
  public void report_not_found() throws Exception {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "/fake/path/report.xml");

    coverageSensor.execute(context);

    // expected logged text: "No report was found for sonar.python.coverage.reportPath using pattern /fake/path/report.xml"
    assertThat(context.lineHits(FILE1_KEY, 1)).isNull();
  }

  @Test
  public void absolute_path() throws Exception {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, new File(moduleBaseDir, "coverage.xml").getAbsolutePath());

    coverageSensor.execute(context);

    assertThat(context.lineHits(FILE1_KEY, 1)).isEqualTo(1);
  }

  @Test
  public void test_coverage() {
    coverageSensor.execute(context);
    Integer[] file1Expected = {1, null, null, 0, null, 0};
    Integer[] file2Expected = {1, 3, 1, 0, 1, 1};

    for (int line = 1; line <= 6; line++) {
      assertThat(context.lineHits(FILE1_KEY, line)).isEqualTo(file1Expected[line - 1]);
      assertThat(context.lineHits(FILE2_KEY, line)).isEqualTo(file2Expected[line - 1]);
      assertThat(context.lineHits(FILE3_KEY, line)).isNull();
      assertThat(context.lineHits(FILE4_KEY, line)).isNull();
    }

    assertThat(context.conditions(FILE2_KEY, 2)).isNull();
    assertThat(context.conditions(FILE2_KEY, 3)).isEqualTo(2);
    assertThat(context.coveredConditions(FILE2_KEY, 3)).isEqualTo(1);
  }

  @Test
  public void test_coverage_4_4_2() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage.4.4.2.xml");
    coverageSensor.execute(context);
    List<Integer> actual = IntStream.range(1, 18).mapToObj(line -> context.lineHits(FILE4_KEY, line)).collect(Collectors.toList());
    assertThat(actual).isEqualTo(Arrays.asList(
      null, // line 1
      null,
      null,
      null,
      null,
      1, // line 6
      1, // line 7
      1, // line 8
      0, // line 9
      1, // line 10
      1, // line 11
      null,
      0, // line 13
      null,
      1, // line 15
      null, // Coverage.py does not consider line 16 and 17 as LOC, here it's null even when "linesOfCode" considers them as code
      null));

    assertThat(context.conditions(FILE4_KEY, 7)).isNull();
    assertThat(context.conditions(FILE4_KEY, 8)).isEqualTo(2);
    assertThat(context.coveredConditions(FILE4_KEY, 8)).isEqualTo(1);
    assertThat(context.conditions(FILE4_KEY, 10)).isEqualTo(2);
    assertThat(context.coveredConditions(FILE4_KEY, 10)).isEqualTo(1);
  }

  @Test
  public void test_coverage_4_4_2_multi_source() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage.4.4.2-multi-sources.xml");
    coverageSensor.execute(context);

    assertThat(context.lineHits("moduleKey:sources/folder1/file1.py", 1)).isEqualTo(1);
    // file2.py ambiguity
    assertThat(context.lineHits("moduleKey:sources/folder1/file2.py", 1)).isNull();
    assertThat(context.lineHits("moduleKey:sources/folder2/file2.py", 1)).isNull();
  }

  @Test
  public void test_unique_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "*coverage.4.4.2*.xml");
    settings.setProperty(PythonCoverageSensor.IT_REPORT_PATH_KEY, "coverage*.4.4.2.xml");
    settings.setProperty(PythonCoverageSensor.OVERALL_REPORT_PATH_KEY, "*coverage.4.4.2.xml");
    coverageSensor.execute(context);
    List<Integer> actual = IntStream.range(1, 18).mapToObj(line -> context.lineHits(FILE4_KEY, line)).collect(Collectors.toList());
    Integer coverageAtLine6 = actual.get(5);
    assertThat(coverageAtLine6).isEqualTo(1);
  }

  @Test
  public void test_unresolved_path() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage_with_unresolved_path.xml");
    coverageSensor.execute(context);

    assertThat(context.lineHits(FILE1_KEY, 1)).isEqualTo(1);
  }

  @Test(expected = IllegalStateException.class)
  public void should_fail_on_invalid_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "invalid-coverage-result.xml");
    coverageSensor.execute(context);
  }

  @Test(expected = IllegalStateException.class)
  public void should_fail_on_unexpected_eof() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage_with_eof_error.xml");
    coverageSensor.execute(context);
  }

  @Test
  public void should_do_nothing_on_empty_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "empty-coverage-result.xml");
    settings.setProperty(PythonCoverageSensor.IT_REPORT_PATH_KEY, "this-file-does-not-exist.xml");
    coverageSensor.execute(context);

    assertThat(context.lineHits(FILE1_KEY, 1)).isNull();
  }

}
