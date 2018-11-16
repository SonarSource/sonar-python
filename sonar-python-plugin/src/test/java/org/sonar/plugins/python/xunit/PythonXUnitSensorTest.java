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
package org.sonar.plugins.python.xunit;

import java.io.File;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.fs.InputComponent;
import org.sonar.api.batch.fs.internal.DefaultFileSystem;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.config.Settings;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.Metric;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonXUnitSensorTest {

  private File baseDir = new File("src/test/resources/org/sonar/plugins/python");
  Settings settings;
  PythonXUnitSensor sensor;
  SensorContextTester context = SensorContextTester.create(baseDir);
  DefaultFileSystem fs;

  @Before
  public void setUp() {
    settings = context.settings();
    fs = new DefaultFileSystem(baseDir);
    sensor = new PythonXUnitSensor(context.config(), fs);
  }

  @Test
  public void shouldSaveCorrectMeasures() {
    DefaultInputFile testFile1 = TestInputFileBuilder.create("", "test_sample1.py").build();
    DefaultInputFile testFile2 = TestInputFileBuilder.create("", "tests/dir/test_sample2.py").build();
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
  public void shouldSaveCorrectMeasuresSimpleMode() {
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
  public void shouldReportNothingWhenNoReportFound() {
    DefaultInputFile testFile1 = TestInputFileBuilder.create("", "test_sample1.py").build();
    fs.add(testFile1);

    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "notexistingpath");
    sensor = new PythonXUnitSensor(context.config(), fs);
    sensor.execute(context);

    assertThat(context.measures(context.module().key())).isEmpty();
    assertThat(context.measures(testFile1.key())).isEmpty();
  }

  @Test(expected = IllegalStateException.class)
  public void shouldThrowWhenGivenInvalidTime() {
    settings.setProperty(PythonXUnitSensor.REPORT_PATH_KEY, "xunit-reports/invalid-time-xunit-report.xml");
    sensor = new PythonXUnitSensor(context.config(), fs);
    sensor.execute(context);
  }

  private Integer moduleMeasure(Metric<Integer> metric) {
    return measure(context.module(), metric);
  }

  private Integer measure(InputComponent component, Metric<Integer> metric) {
    return context.measure(component.key(), metric).value();
  }

}
