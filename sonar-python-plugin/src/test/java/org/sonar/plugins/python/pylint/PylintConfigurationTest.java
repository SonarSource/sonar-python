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
package org.sonar.plugins.python.pylint;

import org.apache.commons.configuration.BaseConfiguration;
import org.apache.commons.configuration.Configuration;
import org.junit.Test;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.ProjectFileSystem;

import java.io.File;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class PylintConfigurationTest {

  @Test
  public void shouldGetCorrectPylintPath() {
    Configuration conf = new BaseConfiguration();
    PylintConfiguration pylintConfiguration = new PylintConfiguration(conf);

    ProjectFileSystem pfs = mock(ProjectFileSystem.class);
    when(pfs.getBasedir()).thenReturn(new File("/projectroot"));
    Project project = new Project("foo");
    project.setFileSystem(pfs);

    assertThat(pylintConfiguration.getPylintConfigPath(project)).isNull();

    conf.setProperty(PylintConfiguration.PYLINT_CONFIG_KEY, "");
    assertThat(pylintConfiguration.getPylintConfigPath(project)).isNull();

    conf.setProperty(PylintConfiguration.PYLINT_CONFIG_KEY, ".pylintrc");
    assertThat(pylintConfiguration.getPylintConfigPath(project)).isEqualTo(new File("/projectroot/.pylintrc").getAbsolutePath());

    String absolutePath = new File("/absolute/.pylintrc").getAbsolutePath();
    conf.setProperty(PylintConfiguration.PYLINT_CONFIG_KEY, absolutePath);
    assertThat(pylintConfiguration.getPylintConfigPath(project)).isEqualTo(absolutePath);
  }

}
