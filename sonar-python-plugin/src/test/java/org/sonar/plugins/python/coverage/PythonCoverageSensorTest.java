/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.api.io.TempDir;
import org.slf4j.event.Level;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.config.internal.MapSettings;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.plugins.python.TestUtils;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

public class PythonCoverageSensorTest {

  private static final String ABSOLUTE_PATH_PLACEHOLDER = "{ABSOLUTE_PATH_PLACEHOLDER}";
  private static final String FILE1_KEY = "moduleKey:sources/file1.py";
  private static final String FILE2_KEY = "moduleKey:sources/file2.py";
  private static final String FILE3_KEY = "moduleKey:sources/file3.py";
  private static final String FILE4_KEY = "moduleKey:sources/file4.py";
  private SensorContextTester context;
  private MapSettings settings;

  private AnalysisWarningsWrapper analysisWarnings;
  private PythonCoverageSensor coverageSensor;
  private File moduleBaseDir = new File("src/test/resources/org/sonar/plugins/python/coverage-reports").getAbsoluteFile();

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @TempDir
  public Path tmpDir;

  @BeforeEach
  public void init() {
    analysisWarnings = spy(AnalysisWarningsWrapper.class);
    coverageSensor = new PythonCoverageSensor(analysisWarnings);
    settings = new MapSettings();
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "coverage.xml");
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
  public void report_not_found() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "/fake/path/report.xml");

    coverageSensor.execute(context);

    // expected logged text: "No report was found for sonar.python.coverage.reportPath using pattern /fake/path/report.xml"
    assertThat(context.lineHits(FILE1_KEY, 1)).isNull();
  }

  @Test
  public void absolute_path() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, new File(moduleBaseDir, "coverage.xml").getAbsolutePath());

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
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "coverage.4.4.2.xml");
    coverageSensor.execute(context);
    List<Integer> actual = IntStream.range(1, 18).mapToObj(line -> context.lineHits(FILE4_KEY, line)).collect(Collectors.toList());
    assertThat(actual).isEqualTo(Arrays.asList(
      // line 1
      null,
      null,
      null,
      null,
      null,
      // line 6
      1,
      // line 7
      1,
      // line 8
      1,
      // line 9
      0,
      // line 10
      1,
      // line 11
      1,
      null,
      // line 13
      0,
      null,
      // line 15
      1,
      // Coverage.py does not consider line 16 and 17 as LOC, here it's null even when "linesOfCode" considers them as code
      null,
      null));

    assertThat(context.conditions(FILE4_KEY, 7)).isNull();
    assertThat(context.conditions(FILE4_KEY, 8)).isEqualTo(2);
    assertThat(context.coveredConditions(FILE4_KEY, 8)).isEqualTo(1);
    assertThat(context.conditions(FILE4_KEY, 10)).isEqualTo(2);
    assertThat(context.coveredConditions(FILE4_KEY, 10)).isEqualTo(1);
  }

  @Test
  public void test_coverage_4_4_2_multi_source() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "coverage.4.4.2-multi-sources.xml");
    coverageSensor.execute(context);

    assertThat(context.lineHits("moduleKey:sources/folder1/file1.py", 1)).isEqualTo(1);
    // file2.py ambiguity
    assertThat(context.lineHits("moduleKey:sources/folder1/file2.py", 1)).isNull();
    assertThat(context.lineHits("moduleKey:sources/folder2/file2.py", 1)).isNull();
  }

  @Test
  public void test_unique_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "*coverage.4.4.2*.xml");
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "*coverage.4.4.2*.xml");
    coverageSensor.execute(context);
    List<Integer> actual = IntStream.range(1, 18).mapToObj(line -> context.lineHits(FILE4_KEY, line)).collect(Collectors.toList());
    Integer coverageAtLine6 = actual.get(5);
    assertThat(coverageAtLine6).isEqualTo(1);
    verify(analysisWarnings, times(1)).addUnique(eq("Property 'sonar.python.coverage.reportPath' has been removed. Please use 'sonar.python.coverage.reportPaths' instead."));
  }

  @Test
  public void test_report_with_absolute_path() throws Exception {
    String reportPath = createReportWithAbsolutePaths();
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, reportPath);

    coverageSensor.execute(context);

    assertThat(context.lineHits(FILE1_KEY, 1)).isEqualTo(1);
    assertThat(context.lineHits(FILE1_KEY, 2)).isEqualTo(3);
    assertThat(context.lineHits(FILE1_KEY, 3)).isEqualTo(1);
    assertThat(context.lineHits(FILE1_KEY, 4)).isEqualTo(0);
    assertThat(context.lineHits(FILE1_KEY, 5)).isEqualTo(1);
    assertThat(context.lineHits(FILE1_KEY, 6)).isEqualTo(1);
  }

  @Test
  public void test_unresolved_path() {
    logTester.clear();
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage_with_unresolved_path.xml");
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "");
    coverageSensor.execute(context);

    String currentFileSeparator = File.separator;

    // no error expected REPORT_PATH_KEY is ignored.
    assertThat(logTester.logs(Level.ERROR)).isEmpty();
    assertThat(logTester.logs(Level.WARN))
      .contains("Property 'sonar.python.coverage.reportPath' has been removed. Please use 'sonar.python.coverage.reportPaths' instead.");
    assertThat(context.lineHits(FILE1_KEY, 1)).isNull();

    logTester.clear();

    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "coverage_with_unresolved_absolute_path.xml");
    coverageSensor.execute(context);

    String expectedLogMessage = String.format(
      "Cannot resolve the file path '%sabsolute%ssources%snot_exist.py' of the coverage report, the file does not exist in all 'source'.",
      currentFileSeparator,
      currentFileSeparator,
      currentFileSeparator);
    assertThat(logTester.logs(Level.ERROR)).containsExactly(
      expectedLogMessage,
      "Cannot resolve 2 file paths, ignoring coverage measures for those files");
  }

  @Test
  public void test_comma_separated_paths() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "coverage.xml,coverage.4.4.2*.xml");
    coverageSensor.execute(context);

    assertThat(context.conditions(FILE2_KEY, 2)).isNull();
    assertThat(context.conditions(FILE2_KEY, 3)).isEqualTo(2);
    assertThat(context.coveredConditions(FILE2_KEY, 3)).isEqualTo(1);

    assertThat(context.conditions(FILE4_KEY, 7)).isNull();
    assertThat(context.conditions(FILE4_KEY, 8)).isEqualTo(2);
    assertThat(context.coveredConditions(FILE4_KEY, 8)).isEqualTo(1);
    assertThat(context.conditions(FILE4_KEY, 10)).isEqualTo(2);
    assertThat(context.coveredConditions(FILE4_KEY, 10)).isEqualTo(1);
  }

  @Test
  public void test_comma_separated_paths_with_deprecated_property() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "");
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage.xml,coverage.4.4.2*.xml");
    coverageSensor.execute(context);

    // old property does not support comma separated list
    assertThat(context.conditions(FILE2_KEY, 2)).isNull();
    assertThat(context.conditions(FILE2_KEY, 3)).isNull();
    assertThat(context.coveredConditions(FILE2_KEY, 3)).isNull();

    assertThat(context.conditions(FILE4_KEY, 7)).isNull();
    assertThat(context.conditions(FILE4_KEY, 8)).isNull();
    assertThat(context.coveredConditions(FILE4_KEY, 8)).isNull();
    assertThat(context.conditions(FILE4_KEY, 10)).isNull();
    assertThat(context.coveredConditions(FILE4_KEY, 10)).isNull();
  }

  @Test
  public void should_fail_on_invalid_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "invalid-coverage-result.xml");
    assertThatThrownBy(() -> coverageSensor.execute(context)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  public void should_fail_on_unexpected_eof() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "coverage_with_eof_error.xml");
    assertThatThrownBy(() -> coverageSensor.execute(context)).isInstanceOf(IllegalStateException.class);
  }

  @Test
  public void should_do_nothing_on_empty_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "empty-coverage-result.xml");
    coverageSensor.execute(context);

    assertThat(context.lineHits(FILE1_KEY, 1)).isNull();
  }

  @Test
  public void should_warn_if_source_is_not_directory() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATHS_KEY, "coverage_source_invalid_directory.xml");
    coverageSensor.execute(context);
    File file = new File("src/test/resources/org/sonar/plugins/python/coverage-reports/sources/file1.py");
    String message = "Invalid directory path in 'source' element: " + file.getPath();
    assertThat(logTester.logs(Level.WARN)).contains(message);
    verify(analysisWarnings, times(1)).addUnique("The following error(s) occurred while trying to import coverage report:" + System.lineSeparator() + message);
  }

  @Test
  public void no_default_report_log() {
    settings.clear();
    PythonCoverageSensor sensor = new PythonCoverageSensor(analysisWarnings);
    sensor.execute(context);
    assertThat(logTester.logs(Level.DEBUG)).contains("No report was found for sonar.python.coverage.reportPaths using default pattern coverage-reports/*coverage-*.xml");
  }

  @Test
  public void sensor_descriptor() {
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();
    new PythonCoverageSensor(analysisWarnings).describe(descriptor);
    assertThat(descriptor.name()).isEqualTo("Cobertura Sensor for Python coverage");
    assertThat(descriptor.languages()).containsOnly("py");
  }

  private String createReportWithAbsolutePaths() throws Exception {
    Path workDir = tmpDir.toAbsolutePath().resolve("python");
    Files.createDirectories(workDir);

    String absoluteSourcePath = new File(moduleBaseDir, "sources/file1.py").getAbsolutePath();
    Path report = new File(moduleBaseDir, "coverage_absolute_path.xml").toPath();
    String reportContent = new String(Files.readAllBytes(report), UTF_8);
    reportContent = reportContent.replace(ABSOLUTE_PATH_PLACEHOLDER, absoluteSourcePath);

    Path reportCopy = workDir.resolve("coverage_absolute_path.xml");
    Files.write(reportCopy, reportContent.getBytes(UTF_8));

    return reportCopy.toAbsolutePath().toString();
  }

}
