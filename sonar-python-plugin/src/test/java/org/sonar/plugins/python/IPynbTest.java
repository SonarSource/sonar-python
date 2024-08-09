/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import org.junit.jupiter.api.Test;
import org.sonar.api.config.internal.ConfigurationBridge;
import org.sonar.api.config.internal.MapSettings;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.PythonPlugin.IPYNB_FILE_SUFFIXES_KEY;

class IPynbTest {

  @Test
  void test() {
    IPynb language = new IPynb(new ConfigurationBridge(new MapSettings()));
    assertThat(language.getKey()).isEqualTo("ipynb");
    assertThat(language.getName()).isEqualTo("IPython Notebooks");
    assertThat(language.getFileSuffixes()).hasSize(1).contains("ipynb");
  }

  @Test
  void custom_file_suffixes() {
    MapSettings settings = new MapSettings();
    settings.setProperty(IPYNB_FILE_SUFFIXES_KEY, "ipynb,my_custom_extension");

    IPynb language = new IPynb(settings.asConfig());
    assertThat(language.getFileSuffixes()).hasSize(2).contains("my_custom_extension").contains("ipynb");
  }

  @Test
  void empty_file_suffixes() {
    MapSettings settings = new MapSettings();
    settings.setProperty(IPYNB_FILE_SUFFIXES_KEY, "");

    IPynb language = new IPynb(settings.asConfig());
    assertThat(language.getFileSuffixes()).hasSize(1).contains("ipynb");
  }
}
