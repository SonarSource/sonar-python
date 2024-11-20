/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python.xunit;

import java.io.File;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.slf4j.event.Level;
import org.sonar.api.batch.fs.InputComponent;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.config.internal.ConfigurationBridge;
import org.sonar.api.config.internal.MapSettings;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.Metric;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class PythonXUnitSensorTest {

  private static final String FILE_SAMPLE1 = "test_sample1.py";
  private static final String FILE_SAMPLE2 = "tests/dir/test_sample2.py";

  private File baseDir = new File("src/test/resources/org/sonar/plugins/python");
  MapSettings settings = new MapSettings();
  PythonXUnitSensor sensor;
  SensorContextTester context = SensorContextTester.create(baseDir);
  DefaultFileSystem fs;
  private final AnalysisWarningsWrapper analysisWarnings = spy(AnalysisWarningsWrapper.class);

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  @BeforeEach
  void setUp() {
    settings = new MapSettings();
    context = SensorContextTester.create(baseDir);
    settings.clear();
    fs = new DefaultFileSystem(baseDir);
    sensor = new PythonXUnitSensor(new ConfigurationBridge(settings), fs, analysisWarnings);
  }

  @Test
  void shouldSaveCorrectMeasures() {
    DefaultInputFile testFile1 = TestInputFileBuilder.create("", FILE_SAMPLE1).build();
    DefaultInputFile testFile2 = TestInputFileBuilder.create("", FILE_SAMPLE2).build();
    fs.add(testFile1);
    fs.add(testFile2);
    sensor.execute(context);

    assertThat(measure(testFile1, CoreMetrics.TESTS)).isEqualTo(3);
    assertThat(measure(testFile2, CoreMetrics.TESTS)).isEqualTo(3);

    assertThat(measure(testFile1, CoreMetrics.SKIPPED_TESTS)).isEqualTo(0);
    assertThat(measure(testFile2, CoreMetrics.SKIPPED_TESTS)).isEqualTo(1);

    assertThat(measure(testFile1, CoreMetrics.TEST_ERRORS)).isEqualTo(1);
    assertThat(measure(testFile2, CoreMetrics.TEST_ERRORS)).isEqualTo(1);

    assertThat(measure(testFile1, CoreMetrics.TEST_FAILURES)).isEqualTo(1);
    assertThat(measure(testFile2, CoreMetrics.TEST_FAILURES)).isEqualTo(1);
  }

  @Test
  void shouldSaveCorrectMeasuresSimpleMode() {
    settings.setProperty(PythonXUnitSensor.SKIP_DETAILS, true);
    fs.add(TestInputFileBuilder.create("", "test_sample.py").build());
    fs.add(TestInputFileBuilder.create("", "tests/dir/test_sample.py").build());
    sensor.execute(context);

    // includes test with not found file
    assertThat(moduleMeasure(CoreMetrics.TESTS)).isEqualTo(7);
    assertThat(moduleMeasure(CoreMetrics.SKIPPED_TESTS)).isEqualTo(1);
    assertThat(moduleMeasure(CoreMetrics.TEST_ERRORS)).isEqualTo(3);
    assertThat(moduleMeasure(CoreMetrics.TEST_FAILURES)).isEqualTo(2);
  }

  @Test
  void shouldReportNothingWhenNoReportFound() {
    DefaultInputFile testFile1 = TestInputFileBuilder.create("", FILE_SAMPLE1).build();
    fs.add(testFile1);

    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "notexistingpath");
    sensor.execute(context);

    assertThat(context.measures(context.module().key())).isEmpty();
    assertThat(context.measures(testFile1.key())).isEmpty();
  }

  @Test
  void shouldLogWarningWhenGivenInvalidTime() {
    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "xunit-reports/invalid-time-xunit-report.xml");
    sensor.execute(context);

    assertThat(logTester.logs(Level.WARN)).contains("Cannot read report 'xunit-reports/invalid-time-xunit-report.xml', " +
      "the following exception occurred: java.text.ParseException: Unparseable number: \"brrrr\"");
    verify(analysisWarnings, times(1))
      .addUnique(eq("An error occurred while trying to import XUnit report(s): 'xunit-reports/invalid-time-xunit-report.xml'"));
  }

  @Test
  void shouldSaveCorrectMeasuresWithPyTestFormat() {
    DefaultInputFile testFile1 = TestInputFileBuilder.create("", FILE_SAMPLE1).build();
    DefaultInputFile testFile2 = TestInputFileBuilder.create("", FILE_SAMPLE2).build();
    fs.add(testFile1);
    fs.add(testFile2);
    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "xunit-reports/pytest-xunit-result.xml");
    sensor.execute(context);

    assertThat(measure(testFile1, CoreMetrics.TESTS)).isEqualTo(2);
    assertThat(measure(testFile2, CoreMetrics.TESTS)).isEqualTo(8);

    assertThat(measure(testFile1, CoreMetrics.SKIPPED_TESTS)).isEqualTo(0);
    assertThat(measure(testFile2, CoreMetrics.SKIPPED_TESTS)).isEqualTo(2);

    assertThat(measure(testFile1, CoreMetrics.TEST_ERRORS)).isEqualTo(0);
    assertThat(measure(testFile2, CoreMetrics.TEST_ERRORS)).isEqualTo(0);

    assertThat(measure(testFile1, CoreMetrics.TEST_FAILURES)).isEqualTo(1);
    assertThat(measure(testFile2, CoreMetrics.TEST_FAILURES)).isEqualTo(1);
  }

  @Test
  void testNoTestReport() {
    DefaultInputFile testFile2 = TestInputFileBuilder.create("", FILE_SAMPLE2).build();
    fs.add(testFile2);
    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "xunit-reports/empty-xunit-result.xml");
    settings.setProperty(PythonXUnitSensor.SKIP_DETAILS, true);
    sensor.execute(context);

    assertThat(context.measure(context.module().key(), CoreMetrics.TESTS)).isNull();
  }

  @Test
  void fallbackToTestsuiteName() {
    DefaultInputFile testFile1 = TestInputFileBuilder.create("", FILE_SAMPLE1).build();
    fs.add(testFile1);
    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "xunit-reports/no-classname-xunit-result.xml");
    sensor.execute(context);

    assertThat(measure(testFile1, CoreMetrics.TESTS)).isEqualTo(2);
    assertThat(measure(testFile1, CoreMetrics.SKIPPED_TESTS)).isEqualTo(0);
    assertThat(measure(testFile1, CoreMetrics.TEST_ERRORS)).isEqualTo(0);
    assertThat(measure(testFile1, CoreMetrics.TEST_FAILURES)).isEqualTo(1);
  }

  @Test
  void malformedReport() {
    DefaultInputFile testFile1 = TestInputFileBuilder.create("", FILE_SAMPLE1).build();
    fs.add(testFile1);
    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "xunit-reports/malformed-xunit-result.xml");
    sensor.execute(context);

    assertThat(measure(testFile1, CoreMetrics.TESTS)).isEqualTo(2);
    assertThat(measure(testFile1, CoreMetrics.SKIPPED_TESTS)).isEqualTo(0);
    assertThat(measure(testFile1, CoreMetrics.TEST_ERRORS)).isEqualTo(0);
    assertThat(measure(testFile1, CoreMetrics.TEST_FAILURES)).isEqualTo(0);
  }

  @Test
  void missingAttributes() {
    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "xunit-reports/missing-attribute-xunit-report.xml");
    sensor.execute(context);
    assertThat(logTester.logs(Level.WARN)).contains("Cannot read report 'xunit-reports/missing-attribute-xunit-report.xml', the following exception occurred: Missing attribute 'time' at line 3");
  }
  private Integer moduleMeasure(Metric<Integer> metric) {
    return measure(context.module(), metric);
  }

  private Integer measure(InputComponent component, Metric<Integer> metric) {
    return context.measure(component.key(), metric).value();
  }

}
