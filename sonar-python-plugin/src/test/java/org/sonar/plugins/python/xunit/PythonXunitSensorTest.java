/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */

package org.sonar.plugins.python.xunit;

import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.CoverageExtension;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.Measure;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.Resource;
import org.sonar.api.scan.filesystem.ModuleFileSystem;
import org.sonar.plugins.python.TestUtils;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Matchers.anyObject;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyDouble;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

public class PythonXunitSensorTest {
  Settings settings;
  PythonXunitSensor sensor;
  SensorContext context;
  Project project;
  ModuleFileSystem fs;

  @Before
  public void setUp() {
    settings = new Settings();
    project = TestUtils.mockProject();
    fs = TestUtils.mockFileSystem();
    sensor = new PythonXunitSensor(settings, TestUtils.mockLanguage(), fs);
    context = mock(SensorContext.class);
  }

  @Test
  public void shouldBeExecutedAfterCoverageExtensions() {
    assertThat(sensor.dependsUponCoverageSensors()).isEqualTo(CoverageExtension.class);
  }

  @Test
  public void shouldSaveCorrectMeasures() {
    sensor.analyse(project, context);

    verify(context, times(4)).saveMeasure((Resource) anyObject(),
      eq(CoreMetrics.TESTS), anyDouble());
    verify(context, times(4)).saveMeasure((Resource) anyObject(),
      eq(CoreMetrics.SKIPPED_TESTS), anyDouble());
    verify(context, times(4)).saveMeasure((Resource) anyObject(),
      eq(CoreMetrics.TEST_ERRORS), anyDouble());
    verify(context, times(4)).saveMeasure((Resource) anyObject(),
      eq(CoreMetrics.TEST_FAILURES), anyDouble());
    verify(context, times(3)).saveMeasure((Resource) anyObject(),
      eq(CoreMetrics.TEST_SUCCESS_DENSITY), anyDouble());
    verify(context, times(4)).saveMeasure((Resource) anyObject(), any(Measure.class));
  }

  @Test
  public void shouldReportNothingWhenNoReportFound() {
    settings.setProperty(PythonXunitSensor.REPORT_PATH_KEY, "notexistingpath");
    sensor = new PythonXunitSensor(settings, TestUtils.mockLanguage(), fs);

    sensor.analyse(project, context);

    verify(context, times(0)).saveMeasure(eq(CoreMetrics.TESTS), any(Double.class));
  }

  @Test(expected = org.sonar.api.utils.SonarException.class)
  public void shouldThrowWhenGivenInvalidTime() {
    settings.setProperty(PythonXunitSensor.REPORT_PATH_KEY, "xunit-reports/invalid-time-xunit-report.xml");
    sensor = new PythonXunitSensor(settings, TestUtils.mockLanguage(), fs);

    sensor.analyse(project, context);
  }
}
