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
package org.sonar.plugins.python;

import com.google.common.collect.Maps;
import java.util.Map;
import org.junit.Test;
import org.sonar.api.config.Settings;
import org.sonar.api.config.internal.MapSettings;

import static org.assertj.core.api.Assertions.assertThat;

public class PythonTest {

  @Test
  public void test() {
    Python language = new Python(new MapSettings());
    assertThat(language.getKey()).isEqualTo("py");
    assertThat(language.getName()).isEqualTo("Python");
    assertThat(language.getFileSuffixes()).hasSize(1).contains("py");
  }

  @Test
  public void custom_file_suffixes() {
    Map<String, String> props = Maps.newHashMap();
    props.put(PythonPlugin.FILE_SUFFIXES_KEY, "py,python");

    Settings settings = new MapSettings();
    settings.addProperties(props);

    Python language = new Python(settings);
    assertThat(language.getFileSuffixes()).hasSize(2).contains("python");
  }
}
