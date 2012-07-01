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

import org.junit.Before;
import org.junit.Test;
import org.sonar.api.profiles.RulesProfile;
import org.sonar.api.resources.Project;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Mockito.mock;

public class PythonSquidSensorTest {

  private PythonSquidSensor sensor;

  @Before
  public void setUp() {
    sensor = new PythonSquidSensor(mock(RulesProfile.class));
  }

  @Test
  public void should_execute_on_python_project() {
    Project project = new Project("key");
    project.setLanguageKey("java");
    assertThat(sensor.shouldExecuteOnProject(project)).isFalse();
    project.setLanguageKey("py");
    assertThat(sensor.shouldExecuteOnProject(project)).isTrue();
  }

}
