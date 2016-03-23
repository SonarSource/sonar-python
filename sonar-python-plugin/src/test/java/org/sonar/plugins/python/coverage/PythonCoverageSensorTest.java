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

import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.Measure;
import org.sonar.api.resources.Project;
import org.sonar.plugins.python.Python;
import org.sonar.api.resources.Resource;
import org.sonar.api.measures.CoreMetrics;

import java.io.File;

import static org.mockito.Matchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import org.mockito.Mockito;

public class PythonCoverageSensorTest {
  PythonCoverageSensor sensor;
  SensorContext context;
  Project project;
  Settings settings;
  DefaultFileSystem fs;

  DefaultInputFile fileWithConditionCoverage, fileWithoutConditionCoverage, fileWithoutCoverageInfo;

  @Before
  public void setUp() {
    project = mock(Project.class);
    settings = new Settings();
    fs = new DefaultFileSystem();
    fs.setBaseDir(new File("src/test/resources/org/sonar/plugins/python"));
    fileWithConditionCoverage = getInputFile("sources/file2.py");
    fileWithoutConditionCoverage = getInputFile("sources/file1.py");
    fileWithoutCoverageInfo = getInputFile("sources/file3.py");
    fs.add(fileWithConditionCoverage);
    fs.add(fileWithoutConditionCoverage);
    fs.add(fileWithoutCoverageInfo);

    context = mock(SensorContext.class);
    Measure measure = new org.sonar.api.measures.Measure();
    measure.setValue(1.0);
    when(context.getMeasure(Mockito.any(Resource.class), Mockito.eq(CoreMetrics.NCLOC))).thenReturn(measure);
  }

  @Test
  public void should_parse_ut_coverage_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage-reports/ut-coverage.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
    verify(context, times(7)).saveMeasure(Mockito.eq(fileWithConditionCoverage), any(Measure.class));
    verify(context, times(3)).saveMeasure(Mockito.eq(fileWithoutConditionCoverage), any(Measure.class));
    verify(context, times(0)).saveMeasure(Mockito.eq(fileWithoutCoverageInfo), any(Measure.class));
  }

  @Test
  public void should_parse_coverage_report_with_zeroing() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage-reports/ut-coverage.xml");
    settings.setProperty(PythonCoverageSensor.FORCE_ZERO_COVERAGE_KEY, true);
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
    // count lineHitsData
    verify(context, times(9)).saveMeasure(Mockito.eq(fileWithConditionCoverage), any(Measure.class));
    verify(context, times(5)).saveMeasure(Mockito.eq(fileWithoutConditionCoverage), any(Measure.class));
    verify(context, times(3)).saveMeasure(Mockito.eq(fileWithoutCoverageInfo), any(Measure.class));
  }

  @Test
  public void should_parse_it_coverage_report() {
    settings.setProperty(PythonCoverageSensor.IT_REPORT_PATH_KEY, "coverage-reports/it-coverage.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
    verify(context, times(7)).saveMeasure(Mockito.eq(fileWithConditionCoverage), any(Measure.class));
    verify(context, times(3)).saveMeasure(Mockito.eq(fileWithoutConditionCoverage), any(Measure.class));
    verify(context, times(0)).saveMeasure(Mockito.eq(fileWithoutCoverageInfo), any(Measure.class));
  }

  @Test
  public void should_parse_overall_coverage_report() {
    settings.setProperty(PythonCoverageSensor.OVERALL_REPORT_PATH_KEY, "coverage-reports/overall-coverage.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
    verify(context, times(7)).saveMeasure(Mockito.eq(fileWithConditionCoverage), any(Measure.class));
    verify(context, times(3)).saveMeasure(Mockito.eq(fileWithoutConditionCoverage), any(Measure.class));
    verify(context, times(0)).saveMeasure(Mockito.eq(fileWithoutCoverageInfo), any(Measure.class));
  }

  @Test(expected = IllegalStateException.class)
  public void shouldFailOnInvalidReport() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage-reports/invalid-coverage-result.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
  }

  @Test(expected = IllegalStateException.class)
  public void shouldFailOnInvalidIntegrationReport() {
    settings.setProperty(PythonCoverageSensor.IT_REPORT_PATH_KEY, "coverage-reports/invalid-coverage-result.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
  }

  @Test
  public void should_do_nothing_on_empty_report() {
    settings.setProperty(PythonCoverageSensor.REPORT_PATH_KEY, "coverage-reports/empty-coverage-result.xml");
    settings.setProperty(PythonCoverageSensor.IT_REPORT_PATH_KEY, "coverage-reports/this-file-does-not-exist.xml");
    sensor = new PythonCoverageSensor(settings, fs);
    sensor.analyse(project, context);
    verify(context, times(0)).saveMeasure(Mockito.eq(fileWithConditionCoverage), any(Measure.class));
    verify(context, times(0)).saveMeasure(Mockito.eq(fileWithoutConditionCoverage), any(Measure.class));
    verify(context, times(0)).saveMeasure(Mockito.eq(fileWithoutCoverageInfo), any(Measure.class));
  }

  private DefaultInputFile getInputFile(String path) {
    DefaultInputFile file = new DefaultInputFile(path);
    file.setLanguage(Python.KEY).setLines(1);
    return file;
  }
}
