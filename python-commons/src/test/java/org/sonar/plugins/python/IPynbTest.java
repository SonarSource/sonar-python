/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.sonar.plugins.python.PythonExtensions.IPYNB_FILE_SUFFIXES_KEY;

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

  @Test
  void equals() {
    ConfigurationBridge emptyConfig = new ConfigurationBridge(new MapSettings());
    IPynb one = new IPynb(emptyConfig);
    IPynb other = new IPynb(emptyConfig);
    IPynb nullConfig = null;

    MapSettings settings = new MapSettings();
    settings.setProperty(IPYNB_FILE_SUFFIXES_KEY, "ipynb,my_custom_extension");
    IPynb customConfig = new IPynb(settings.asConfig());

    assertEquals(one, one);
    assertEquals(one, other);
    assertNotEquals(one, nullConfig);
    assertNotEquals(one, customConfig);
    assertNotEquals(one, new Python(new ConfigurationBridge(new MapSettings())));
  }

  @Test
  void hashcode() {
    ConfigurationBridge emptyConfig = new ConfigurationBridge(new MapSettings());
    IPynb one = new IPynb(emptyConfig);
    IPynb other = new IPynb(emptyConfig);
    assertEquals(one.hashCode(), other.hashCode());
    assertNotEquals(one.hashCode(), "test".hashCode());
  }
}
