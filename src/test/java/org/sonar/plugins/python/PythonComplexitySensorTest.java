/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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

import static org.mockito.Matchers.anyObject;
import static org.mockito.Matchers.argThat;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;

import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.resources.ProjectFileSystem;
import org.sonar.api.resources.Resource;
import org.sonar.api.test.IsMeasure;

public class PythonComplexitySensorTest {

  private PythonComplexitySensor sensor;
  private SensorContext context;
  private ProjectFileSystem fileSystem;

  @Before
  public void init() {
    sensor = new PythonComplexitySensor();
    context = mock(SensorContext.class);
    fileSystem = mock(ProjectFileSystem.class);
  }

  @Test
  public void testComplexityMeasures() throws URISyntaxException, IOException {
    String resourceName = "/org/sonar/plugins/python/complexity/code_chunks.py";
    File file = new File(getClass().getResource(resourceName).toURI());
    sensor.analyzeFile(file, fileSystem, context);

    verify(context).saveMeasure((Resource) anyObject(), eq(CoreMetrics.COMPLEXITY), eq(47.0));
    verify(context).saveMeasure((Resource) anyObject(),
        argThat(new IsMeasure(CoreMetrics.FILE_COMPLEXITY_DISTRIBUTION, "0=0;5=0;10=0;20=0;30=1;60=0;90=0")));
    // verify(context).saveMeasure((Resource) anyObject(),
    // eq(CoreMetrics.FUNCTION_COMPLEXITY),
    // eq(3.285714286));
    verify(context).saveMeasure((Resource) anyObject(),
        argThat(new IsMeasure(CoreMetrics.FUNCTION_COMPLEXITY_DISTRIBUTION, "1=0;2=9;4=5;6=0;8=0;10=0;12=0;20=0;30=0")));
  }
}
