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
package org.sonar.plugins.python.pylint;

import org.junit.Before;
import org.junit.Test;
import org.sonar.api.config.Settings;
import org.sonar.api.scan.filesystem.ModuleFileSystem;

import java.io.File;

import static org.fest.assertions.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

public class PylintConfigurationTest {

  private Settings settings;
  private PylintConfiguration pylintConfiguration;

  @Before
  public void setUp() throws Exception {
    settings = new Settings();
    pylintConfiguration = new PylintConfiguration(settings);
  }

  @Test
  public void shouldGetCorrectPylintPath() {
    ModuleFileSystem fs = mock(ModuleFileSystem.class);
    when(fs.baseDir()).thenReturn(new File("/projectroot"));

    assertThat(pylintConfiguration.getPylintConfigPath(fs)).isNull();

    settings.setProperty(PylintConfiguration.PYLINT_CONFIG_KEY, (String) null);
    assertThat(pylintConfiguration.getPylintConfigPath(fs)).isNull();

    settings.setProperty(PylintConfiguration.PYLINT_CONFIG_KEY, ".pylintrc");
    assertThat(pylintConfiguration.getPylintConfigPath(fs)).isEqualTo(new File("/projectroot/.pylintrc").getAbsolutePath());

    String absolutePath = new File("/absolute/.pylintrc").getAbsolutePath();
    settings.setProperty(PylintConfiguration.PYLINT_CONFIG_KEY, absolutePath);
    assertThat(pylintConfiguration.getPylintConfigPath(fs)).isEqualTo(absolutePath);
  }

  @Test
  public void getPylintPath() {
    String path = "test/path";
    settings.setProperty(PylintConfiguration.PYLINT_KEY, path);

    assertThat(pylintConfiguration.getPylintPath()).isEqualTo(path);
  }
}
