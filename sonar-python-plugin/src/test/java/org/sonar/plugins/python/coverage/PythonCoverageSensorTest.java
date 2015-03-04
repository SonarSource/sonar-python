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

import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.Measure;
import org.sonar.api.resources.Project;

import java.io.File;

import static org.mockito.Matchers.any;
import static org.mockito.Matchers.anyObject;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

public class PythonCoverageSensorTest {
  PythonCoverageSensor sensor;
  SensorContext context;
  Project project;
  Settings settings;
  DefaultFileSystem fs;

  @Before
  public void setUp() {
    project = mock(Project.class);
    settings = new Settings();
    fs = new DefaultFileSystem();
    fs.setBaseDir(new File("src/test/resources/org/sonar/plugins/python"));
    sensor = new PythonCoverageSensor(settings, fs);
    context = mock(SensorContext.class);
  }

  @Test
  public void shouldReportCorrectIssues() {
    fs.add(new DefaultInputFile("sources/file1.py"));
    fs.add(new DefaultInputFile("sources/file2.py"));
    fs.add(new DefaultInputFile("sources/file3.py"));
    fs.add(new DefaultInputFile("sources/file4.py"));
    fs.add(new DefaultInputFile("sources/file5.py"));
    fs.add(new DefaultInputFile("sources/file6.py"));
    fs.add(new DefaultInputFile("sources/file7.py"));
    sensor.analyse(project, context);
    verify(context, times(33)).saveMeasure((InputFile) anyObject(), any(Measure.class));
  }

  @Test(expected=IllegalStateException.class)
  public void shouldFailOnInvalidReport() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage-reports/invalid-coverage-result.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
  }

  @Test(expected=IllegalStateException.class)
  public void shouldFailOnInvalidIntegrationReport() {
    settings.setProperty(PythonCoverageSensor.IT_REPORT_PATH_KEY, "coverage-reports/invalid-coverage-result.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
  }
}
