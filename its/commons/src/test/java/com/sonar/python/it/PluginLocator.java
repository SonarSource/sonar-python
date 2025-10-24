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

  record LocalPlugin(String localLocation, String localWildcard) {
  }

  public enum Plugins {
    PYTHON(
      new LocalPlugin("../../../sonar-python-plugin/target", "sonar-python-plugin-*.jar"),
      new LocalPlugin("../../sonar-python-enterprise-plugin/target", "sonar-python-enterprise-plugin-*.jar")),
    PYTHON_CUSTOM_RULES(
      new LocalPlugin("../python-custom-rules-plugin/target", "python-custom-rules-plugin-*.jar"),
      new LocalPlugin("../../../its/plugin/python-custom-rules-plugin/target", "python-custom-rules-plugin-*.jar")),
    PYTHON_CUSTOM_RULES_EXAMPLE(
      new LocalPlugin("../../docs/python-custom-rules-example/target", "python-custom-rules-example-*.jar"),
      new LocalPlugin("../../../docs/python-custom-rules-example/target", "python-custom-rules-example-*.jar"));

    private final LocalPlugin ossPlugin;
    private final LocalPlugin enterprisePlugin;

    Plugins(LocalPlugin ossPlugin, LocalPlugin enterprisePlugin) {
      this.ossPlugin = ossPlugin;
      this.enterprisePlugin = enterprisePlugin;
    }

    public Location get(boolean useEnterprise) {
      LocalPlugin plugin = useEnterprise ? enterprisePlugin : ossPlugin;
      return FileLocation.byWildcardMavenFilename(new File(plugin.localLocation()).getAbsoluteFile(), plugin.localWildcard());
    }
  }

  public static boolean isEnterpriseTest() {
    var currentWorkingDirectory = new File(System.getProperty("user.dir"));
    return currentWorkingDirectory.getParentFile().getName().equals("its-enterprise");
  }

  private PluginLocator() {
    // utility class
  }
}
