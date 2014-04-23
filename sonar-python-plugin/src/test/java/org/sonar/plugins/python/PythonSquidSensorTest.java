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
package org.sonar.plugins.python;

import com.google.common.collect.ImmutableList;
import org.apache.commons.collections.ListUtils;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.component.ResourcePerspectives;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.profiles.RulesProfile;
import org.sonar.api.resources.*;
import org.sonar.api.scan.filesystem.FileQuery;
import org.sonar.api.scan.filesystem.ModuleFileSystem;

import java.io.File;
import java.nio.charset.Charset;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class PythonSquidSensorTest {

  private FileLinesContextFactory fileLinesContextFactory;

  @Before
  public void setUp() {
    fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(Resource.class))).thenReturn(fileLinesContext);
  }

  @Test
  public void should_execute_on_python_project() {
    Project project = mock(Project.class);
    ModuleFileSystem fs = mock(ModuleFileSystem.class);
    PythonSquidSensor sensor = new PythonSquidSensor(mock(RulesProfile.class), fileLinesContextFactory, fs, mock(ResourcePerspectives.class));

    when(fs.files(any(FileQuery.class))).thenReturn(ListUtils.EMPTY_LIST);
    assertThat(sensor.shouldExecuteOnProject(project)).isFalse();

    when(fs.files(any(FileQuery.class))).thenReturn(ImmutableList.of(new File("/tmp")));
    assertThat(sensor.shouldExecuteOnProject(project)).isTrue();
  }

  @Test
  public void should_analyse() {
    ModuleFileSystem fs = mock(ModuleFileSystem.class);
    when(fs.sourceCharset()).thenReturn(Charset.forName("UTF-8"));
    when(fs.files(any(FileQuery.class))).thenReturn(ImmutableList.of(
      new File("src/test/resources/org/sonar/plugins/python/code_chunks_2.py")));

    ProjectFileSystem pfs = mock(ProjectFileSystem.class);
    when(pfs.getSourceDirs()).thenReturn(ImmutableList.of(new File("src/test/resources/org/sonar/plugins/python/")));

    Project project = new Project("key");
    project.setFileSystem(pfs);
    SensorContext context = mock(SensorContext.class);
    PythonSquidSensor sensor = new PythonSquidSensor(mock(RulesProfile.class), fileLinesContextFactory, fs, mock(ResourcePerspectives.class));

    sensor.analyse(project, context);

    verify(context).saveMeasure(Mockito.any(Resource.class), Mockito.eq(CoreMetrics.FILES), Mockito.eq(1.0));
    verify(context).saveMeasure(Mockito.any(Resource.class), Mockito.eq(CoreMetrics.LINES), Mockito.eq(29.0));
    verify(context).saveMeasure(Mockito.any(Resource.class), Mockito.eq(CoreMetrics.NCLOC), Mockito.eq(25.0));
    verify(context).saveMeasure(Mockito.any(Resource.class), Mockito.eq(CoreMetrics.STATEMENTS), Mockito.eq(23.0));
    verify(context).saveMeasure(Mockito.any(Resource.class), Mockito.eq(CoreMetrics.FUNCTIONS), Mockito.eq(4.0));
    verify(context).saveMeasure(Mockito.any(Resource.class), Mockito.eq(CoreMetrics.CLASSES), Mockito.eq(1.0));
    verify(context).saveMeasure(Mockito.any(Resource.class), Mockito.eq(CoreMetrics.COMPLEXITY), Mockito.eq(4.0));
    verify(context).saveMeasure(Mockito.any(Resource.class), Mockito.eq(CoreMetrics.COMMENT_LINES), Mockito.eq(9.0));
  }

  @Test
  public void test_toString() {
    PythonSquidSensor sensor = new PythonSquidSensor(mock(RulesProfile.class), fileLinesContextFactory, null, mock(ResourcePerspectives.class));
    assertThat(sensor.toString()).isEqualTo("PythonSquidSensor");
  }

}
