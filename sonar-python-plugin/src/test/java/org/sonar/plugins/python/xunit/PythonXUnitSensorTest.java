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
package org.sonar.plugins.python.xunit;

import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.CoverageExtension;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.Measure;
import org.sonar.api.resources.Project;

import java.io.File;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyDouble;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

public class PythonXUnitSensorTest {
  Settings settings;
  PythonXUnitSensor sensor;
  SensorContext context;
  Project project;
  DefaultFileSystem fs;

  @Before
  public void setUp() {
    settings = new Settings();
    project = mock(Project.class);
    fs = new DefaultFileSystem();
    fs.setBaseDir(new File("src/test/resources/org/sonar/plugins/python"));
    context = mock(SensorContext.class);
    sensor = new PythonXUnitSensor(settings, fs);
  }

  @Test
  public void shouldBeExecutedAfterCoverageExtensions() {
    assertThat(sensor.dependsUponCoverageSensors()).isEqualTo(CoverageExtension.class);
  }

  @Test
  public void shouldSaveCorrectMeasures() {
    DefaultInputFile testFile1 = new DefaultInputFile("test_sample1.py");
    DefaultInputFile testFile2 = new DefaultInputFile("tests/dir/test_sample2.py");
    fs.add(testFile1);
    fs.add(testFile2);
    sensor.analyse(project, context);

    verify(context).saveMeasure(testFile1, CoreMetrics.TESTS, 3.);
    verify(context).saveMeasure(testFile2, CoreMetrics.TESTS, 3.);
    verify(context).saveMeasure(testFile1, CoreMetrics.TESTS, 0.);

    verify(context).saveMeasure(testFile1, CoreMetrics.SKIPPED_TESTS, 0.);
    verify(context).saveMeasure(testFile2, CoreMetrics.SKIPPED_TESTS, 1.);
    verify(context).saveMeasure(testFile1, CoreMetrics.SKIPPED_TESTS, 1.);

    verify(context).saveMeasure(testFile1, CoreMetrics.TEST_ERRORS, 1.);
    verify(context).saveMeasure(testFile2, CoreMetrics.TEST_ERRORS, 1.);
    verify(context).saveMeasure(testFile1, CoreMetrics.TEST_ERRORS, 0.);

    verify(context).saveMeasure(testFile1, CoreMetrics.TEST_FAILURES, 1.);
    verify(context).saveMeasure(testFile2, CoreMetrics.TEST_FAILURES, 1.);
    verify(context).saveMeasure(testFile1, CoreMetrics.TEST_FAILURES, 0.);

    verify(context).saveMeasure(eq(testFile1), eq(CoreMetrics.TEST_SUCCESS_DENSITY), anyDouble());
    verify(context).saveMeasure(eq(testFile2), eq(CoreMetrics.TEST_SUCCESS_DENSITY), anyDouble());

    verify(context, times(2)).saveMeasure(eq(testFile1), any(Measure.class));
    verify(context).saveMeasure(eq(testFile2), any(Measure.class));
  }

  @Test
  public void shouldSaveCorrectMeasuresSimpleMode() {
    settings.setProperty(PythonXUnitSensor.SKIP_DETAILS, true);
    fs.add(new DefaultInputFile("test_sample.py"));
    fs.add(new DefaultInputFile("tests/dir/test_sample.py"));
    sensor.analyse(project, context);

    verify(context).saveMeasure(eq(CoreMetrics.TEST_SUCCESS_DENSITY), anyDouble());
    // includes test with not found file
    verify(context).saveMeasure(CoreMetrics.TESTS, 7.);
    verify(context).saveMeasure(CoreMetrics.SKIPPED_TESTS, 2.);
    // includes test with not found file
    verify(context).saveMeasure(CoreMetrics.TEST_ERRORS, 3.);
    verify(context).saveMeasure(CoreMetrics.TEST_FAILURES, 2.);
    verify(context).saveMeasure(eq(CoreMetrics.TEST_EXECUTION_TIME), anyDouble());
    verifyNoMoreInteractions(context);
  }

  @Test
  public void shouldReportNothingWhenNoReportFound() {
    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "notexistingpath");
    sensor = new PythonXUnitSensor(settings, fs);
    sensor.analyse(project, context);

    verifyNoMoreInteractions(context);
  }

  @Test(expected = IllegalStateException.class)
  public void shouldThrowWhenGivenInvalidTime() {
    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "xunit-reports/invalid-time-xunit-report.xml");
    sensor = new PythonXUnitSensor(settings, fs);
    sensor.analyse(project, context);
  }
}
