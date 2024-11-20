/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python;

import org.junit.jupiter.api.Test;
import org.sonar.api.config.internal.ConfigurationBridge;
import org.sonar.api.config.internal.MapSettings;

import static org.assertj.core.api.Assertions.assertThat;

class PythonTest {

  @Test
  void test() {
    Python language = new Python(new ConfigurationBridge(new MapSettings()));
    assertThat(language.getKey()).isEqualTo("py");
    assertThat(language.getName()).isEqualTo("Python");
    assertThat(language.getFileSuffixes()).hasSize(1).contains("py");
  }

  @Test
  void custom_file_suffixes() {
    MapSettings settings = new MapSettings();
    settings.setProperty(PythonPlugin.PYTHON_FILE_SUFFIXES_KEY, "py,python");

    Python language = new Python(new ConfigurationBridge(settings));
    assertThat(language.getFileSuffixes()).hasSize(2).contains("python");
  }
}
