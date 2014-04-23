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
package org.sonar.plugins.python.coverage;

import static org.mockito.Matchers.any;
import static org.mockito.Matchers.anyObject;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.Measure;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.Resource;
import org.sonar.api.scan.filesystem.ModuleFileSystem;
import org.sonar.plugins.python.TestUtils;

public class PythonCoverageSensorTest {
  PythonCoverageSensor sensor;
  SensorContext context;
  Project project;
  Settings settings;
  ModuleFileSystem fs;

  @Before
  public void setUp() {
    project = TestUtils.mockProject();
    settings = new Settings();
    fs = TestUtils.mockFileSystem();
    sensor = new PythonCoverageSensor(settings, fs);
    context = mock(SensorContext.class);
    Resource resourceMock = mock(Resource.class);
    when(context.getResource((Resource)anyObject())).thenReturn(resourceMock);
  }

  @Test
  public void shouldReportCorrectIssues() {
    sensor.analyse(project, context);
    verify(context, times(66)).saveMeasure((Resource) anyObject(), any(Measure.class));
  }

  @Test(expected=org.sonar.api.utils.SonarException.class)
  public void shouldFailOnInvalidReport() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage-reports/invalid-coverage-result.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
  }

  @Test(expected=org.sonar.api.utils.SonarException.class)
  public void shouldFailOnInvalidIntegrationReport() {
    settings.setProperty(PythonCoverageSensor.IT_REPORT_PATH_KEY, "coverage-reports/invalid-coverage-result.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
  }
}
