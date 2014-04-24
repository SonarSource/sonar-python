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
package org.sonar.plugins.python;

import com.google.common.collect.ImmutableList;
import org.sonar.api.SonarPlugin;
import org.sonar.api.config.PropertyDefinition;
import org.sonar.api.resources.Qualifiers;
import org.sonar.plugins.python.colorizer.PythonColorizer;
import org.sonar.plugins.python.coverage.PythonCoverageSensor;
import org.sonar.plugins.python.cpd.PythonCpdMapping;
import org.sonar.plugins.python.pylint.PylintConfiguration;
import org.sonar.plugins.python.pylint.PylintRuleRepository;
import org.sonar.plugins.python.pylint.PylintSensor;
import org.sonar.plugins.python.xunit.PythonXunitSensor;

import java.util.List;

public class PythonPlugin extends SonarPlugin {

  public static final String FILE_SUFFIXES_KEY = "sonar.python.file.suffixes";

  public List getExtensions() {
    return ImmutableList.of(

        PropertyDefinition.builder(FILE_SUFFIXES_KEY)
          .name("File Suffixes")
          .description("Comma-separated list of suffixes of Python files to analyze.")
          .category("Python")
          .onQualifiers(Qualifiers.PROJECT)
          .defaultValue("py")
          .build(),

        Python.class,
        PythonSourceImporter.class,
        PythonColorizer.class,
        PythonCpdMapping.class,

        PythonSquidSensor.class,
        PythonRuleRepository.class,
        PythonDefaultProfile.class,

        PythonCommonRulesEngine.class,

        // pylint
        PylintConfiguration.class,
        PylintSensor.class,
        PylintRuleRepository.class,

        PythonXunitSensor.class,
        PythonCoverageSensor.class);
  }

}
