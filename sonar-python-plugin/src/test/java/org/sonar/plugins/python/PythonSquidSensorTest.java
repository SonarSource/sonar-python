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
package org.sonar.plugins.python;

import java.io.File;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.FileMetadata;
import org.sonar.api.batch.rule.ActiveRules;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.internal.ActiveRulesBuilder;
import org.sonar.api.batch.sensor.internal.DefaultSensorDescriptor;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.internal.google.common.base.Charsets;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.rule.RuleKey;
import org.sonar.python.checks.CheckList;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class PythonSquidSensorTest {

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python/");
  private SensorContextTester context = SensorContextTester.create(baseDir);
  private PythonSquidSensor sensor;

  @Before
  public void setUp() {
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(InputFile.class))).thenReturn(fileLinesContext);
    ActiveRules activeRules = (new ActiveRulesBuilder())
        .create(RuleKey.of(CheckList.REPOSITORY_KEY, "PrintStatementUsage"))
        .setName("Print Statement Usage")
        .activate()
        .build();
    CheckFactory checkFactory = new CheckFactory(activeRules);
    sensor = new PythonSquidSensor(fileLinesContextFactory, checkFactory);
  }

  @Test
  public void sensor_descriptor() {
    DefaultSensorDescriptor descriptor = new DefaultSensorDescriptor();

    sensor.describe(descriptor);
    assertThat(descriptor.name()).isEqualTo("Python Squid Sensor");
    assertThat(descriptor.languages()).containsOnly("py");
    assertThat(descriptor.type()).isEqualTo(Type.MAIN);
  }



  @Test
  public void test_execute() {
    DefaultInputFile inputFile = new DefaultInputFile("moduleKey", "code_chunks_2.py")
      .setModuleBaseDir(baseDir.toPath())
      .setType(Type.MAIN)
      .setLanguage(Python.KEY);
    context.fileSystem().add(inputFile);
    inputFile.initMetadata(new FileMetadata().readMetadata(inputFile.file(), Charsets.UTF_8));

    sensor.execute(context);

    String key = "moduleKey:code_chunks_2.py";
    assertThat(context.measure(key, CoreMetrics.FILES).value()).isEqualTo(1);
    assertThat(context.measure(key, CoreMetrics.LINES).value()).isEqualTo(29);
    assertThat(context.measure(key, CoreMetrics.NCLOC).value()).isEqualTo(25);
    assertThat(context.measure(key, CoreMetrics.STATEMENTS).value()).isEqualTo(23);
    assertThat(context.measure(key, CoreMetrics.FUNCTIONS).value()).isEqualTo(4);
    assertThat(context.measure(key, CoreMetrics.CLASSES).value()).isEqualTo(1);
    assertThat(context.measure(key, CoreMetrics.COMPLEXITY).value()).isEqualTo(4);
    assertThat(context.measure(key, CoreMetrics.COMMENT_LINES).value()).isEqualTo(9);

    assertThat(context.allIssues()).hasSize(1);
  }

}
