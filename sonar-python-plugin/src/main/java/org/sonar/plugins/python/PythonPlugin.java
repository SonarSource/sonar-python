/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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

import com.google.common.collect.ImmutableList;
import java.util.List;
import org.sonar.api.PropertyType;
import org.sonar.api.SonarPlugin;
import org.sonar.api.config.PropertyDefinition;
import org.sonar.api.resources.Qualifiers;
import org.sonar.plugins.python.coverage.PythonCoverageSensor;
import org.sonar.plugins.python.cpd.PythonCpdMapping;
import org.sonar.plugins.python.pylint.PylintConfiguration;
import org.sonar.plugins.python.pylint.PylintImportSensor;
import org.sonar.plugins.python.pylint.PylintRuleRepository;
import org.sonar.plugins.python.pylint.PylintSensor;
import org.sonar.plugins.python.xunit.PythonXUnitSensor;

public class PythonPlugin extends SonarPlugin {

  public static final String FILE_SUFFIXES_KEY = "sonar.python.file.suffixes";
  private static final String CATEGORY = "Python";

  @Override
  public List getExtensions() {

    return ImmutableList.of(

      PropertyDefinition.builder(FILE_SUFFIXES_KEY)
        .name("File Suffixes")
        .description("Comma-separated list of suffixes of Python files to analyze.")
        .category(CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("py")
        .build(),

      PropertyDefinition.builder(PythonCoverageSensor.REPORT_PATH_KEY)
        .name("Path to coverage report(s)")
        .description("Path to coverage reports. Ant patterns are accepted for relative path. The reports have to conform to the Cobertura XML format.")
        .category(CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue(PythonCoverageSensor.DEFAULT_REPORT_PATH)
        .build(),
      PropertyDefinition.builder(PythonCoverageSensor.IT_REPORT_PATH_KEY)
        .name("Path to coverage report(s) for integration tests")
        .description("Path to coverage reports for integration tests. Ant patterns are accepted for relative path. The reports have to conform to the Cobertura XML format.")
        .category(CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue(PythonCoverageSensor.IT_DEFAULT_REPORT_PATH)
        .build(),
      PropertyDefinition.builder(PythonCoverageSensor.OVERALL_REPORT_PATH_KEY)
        .name("Path to overall (combined UT+IT) coverage report(s)")
        .description("Path to a report containing overall test coverage data (i.e. test coverage gained by all tests of all kinds). " +
          "Ant patterns are accepted for relative path. The reports have to conform to the Cobertura XML format.")
        .category(CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue(PythonCoverageSensor.OVERALL_DEFAULT_REPORT_PATH)
        .build(),

      PropertyDefinition.builder(PythonCoverageSensor.FORCE_ZERO_COVERAGE_KEY)
        .name("Assign zero line coverage to source files without coverage report(s)")
        .description("If 'True', assign zero line coverage to source files without coverage report(s), which results in a more realistic overall Technical Debt value.")
        .category(CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("false")
        .type(PropertyType.BOOLEAN)
        .build(),

      Python.class,
      PythonCpdMapping.class,

      PythonProfile.class,

      PythonSquidSensor.class,
      PythonRuleRepository.class,

      PylintConfiguration.class,
      PylintSensor.class,
      PylintImportSensor.class,
      PylintRuleRepository.class,

      PythonXUnitSensor.class);
  }

}
