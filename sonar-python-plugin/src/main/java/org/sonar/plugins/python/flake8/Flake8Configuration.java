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
package org.sonar.plugins.python.flake8;

import org.apache.commons.lang.StringUtils;
import org.sonar.api.BatchExtension;
import org.sonar.api.Properties;
import org.sonar.api.Property;
import org.sonar.api.config.Settings;
import org.sonar.api.scan.filesystem.ModuleFileSystem;

import java.io.File;

@Properties({
  @Property(
    key = Flake8Configuration.FLAKE8_CONFIG_KEY,
    defaultValue = "",
    name = "flake8 configuration",
    description = "Path to the flake8 configuration file to use in flake8 analysis. Set to empty to use the default.",
    global = false,
    project = true),
  @Property(
    key = Flake8Configuration.FLAKE8_KEY,
    defaultValue = "",
    name = "flake8 executable",
    description = "Path to the flake8 executable to use in flake8 analysis. Set to empty to use the default one.",
    global = true,
    project = false)
})
public class Flake8Configuration implements BatchExtension {

  public static final String FLAKE8_CONFIG_KEY = "sonar.python.flake8_config";
  public static final String FLAKE8_KEY = "sonar.python.flake8";

  private final Settings conf;

  public Flake8Configuration(Settings conf) {
    this.conf = conf;
  }

  public String getFlake8ConfigPath(ModuleFileSystem fileSystem) {
    String configPath = conf.getString(Flake8Configuration.FLAKE8_CONFIG_KEY);
    if (StringUtils.isEmpty(configPath)) {
      return null;
    }
    File configFile = new File(configPath);
    if (!configFile.isAbsolute()) {
      File projectRoot = fileSystem.baseDir();
      configFile = new File(projectRoot.getPath(), configPath);
    }
    return configFile.getAbsolutePath();
  }

  public String getFlake8Path() {
    return conf.getString(Flake8Configuration.FLAKE8_KEY);
  }

}
