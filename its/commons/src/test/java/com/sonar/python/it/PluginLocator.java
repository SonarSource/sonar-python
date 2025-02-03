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
package com.sonar.python.it;

import com.sonar.orchestrator.locator.FileLocation;
import com.sonar.orchestrator.locator.Location;
import java.io.File;

public class PluginLocator {

  public static final String SQ_PLUGIN_TARGET_PROPERTY = "sonar.targetPlugin";
  public static final String DEFAULT_PLUGIN_TARGET = "OPEN_SOURCE";

  record Plugin(String localLocation, String localWildcard) {
  }

  public enum Plugins {
    PYTHON_OSS("../../../sonar-python-plugin/target", "sonar-python-plugin-*.jar"),
    PYTHON_ENTERPRISE("../../sonar-python-enterprise-plugin/target",
      "sonar-python-enterprise-plugin-*.jar");

    private final Plugin plugin;

    Plugins(String localLocation, String localWildcard) {
      this.plugin = new Plugin(localLocation, localWildcard);
    }

    public Location get() {
      return FileLocation.byWildcardMavenFilename(new File(plugin.localLocation), plugin.localWildcard);
    }
  }

  public static Plugins pythonPlugin() {
    var targetPlugin = System.getProperty(SQ_PLUGIN_TARGET_PROPERTY, DEFAULT_PLUGIN_TARGET);
    if (DEFAULT_PLUGIN_TARGET.equals(targetPlugin)) {
      return Plugins.PYTHON_OSS;
    } else {
      return Plugins.PYTHON_ENTERPRISE;
    }
  }

  public static Location pythonPluginLocation() {
    return pythonPlugin().get();
  }

  private PluginLocator() {
    // utility class
  }
}
