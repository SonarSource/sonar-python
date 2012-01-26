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

import static org.junit.Assert.assertEquals;

import static org.mockito.Matchers.anyObject;
import static org.mockito.Matchers.argThat;
import static org.mockito.Matchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.io.File;

import org.sonar.api.resources.Project;
import org.sonar.api.resources.ProjectFileSystem;
import org.apache.commons.configuration.Configuration;
import org.sonar.api.rules.RuleFinder;

import org.junit.Test;
import org.junit.Before;


public class PythonViolationsSensorTest {
  private PythonViolationsSensor sensor;
  private Project project;
  private ProjectFileSystem pfs;
  private RuleFinder ruleFinder;
  private Configuration conf;
  
  @Before
  public void init() {
    ruleFinder = mock(RuleFinder.class);
    conf = mock(Configuration.class);

    pfs = mock(ProjectFileSystem.class);
    when(pfs.getBasedir()).thenReturn(new File("/tmp"));
    
    project = mock(Project.class);
    when(project.getProperty("sonar.python.path")).thenReturn("path1, path2");
    when(project.getFileSystem()).thenReturn(pfs);
  }
  
  @Test
  public void shouldReturnCorrectEnvironment() {
    sensor = new PythonViolationsSensor(ruleFinder, project, conf);
    String[] env = sensor.getEnvironment(project);
    
    String[] expectedEnv = {"PYTHONPATH=/tmp/path1:/tmp/path2"};
    assertEquals(env, expectedEnv);
  }
}
