/*
 * Sonar Python Plugin
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

import org.apache.commons.configuration.Configuration;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.measures.Measure;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.Resource;
import org.sonar.plugins.python.TestUtils;

public class PythonCoverageSensorTest {
  private PythonCoverageSensor sensor;
  private SensorContext context;
  private Project project;

  @Before
  public void setUp() {
    project = TestUtils.mockProject();
    sensor = new PythonCoverageSensor(mock(Configuration.class));
    context = mock(SensorContext.class);
    Resource resourceMock = mock(Resource.class);
    when(context.getResource((Resource)anyObject())).thenReturn(resourceMock);
  }

  @Test
  public void shouldReportCorrectViolations() {
    sensor.analyse(project, context);
    verify(context, times(66)).saveMeasure((Resource) anyObject(), any(Measure.class));
  }

  @Test(expected=org.sonar.api.utils.SonarException.class)
  public void shouldFailOnInvalidReport() {
    Configuration config = mock(Configuration.class);
    when(config.getString(PythonCoverageSensor.REPORT_PATH_KEY, null))
      .thenReturn("coverage-reports/invalid-coverage-result.xml");
    sensor = new PythonCoverageSensor(config);
    sensor.analyse(project, context);
  }

  @Test(expected=org.sonar.api.utils.SonarException.class)
  public void shouldFailOnInvalidIntegrationReport() {
    Configuration config = mock(Configuration.class);
    when(config.getString(PythonCoverageSensor.IT_REPORT_PATH_KEY, null))
      .thenReturn("coverage-reports/invalid-coverage-result.xml");
    sensor = new PythonCoverageSensor(config);
    sensor.analyse(project, context);
  }
}
