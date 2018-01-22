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

import org.sonar.api.Plugin;
import org.sonar.api.PropertyType;
import org.sonar.api.SonarProduct;
import org.sonar.api.config.PropertyDefinition;
import org.sonar.api.resources.Qualifiers;
import org.sonar.plugins.python.coverage.PythonCoverageSensor;
import org.sonar.plugins.python.pylint.PylintConfiguration;
import org.sonar.plugins.python.pylint.PylintImportSensor;
import org.sonar.plugins.python.pylint.PylintRuleRepository;
import org.sonar.plugins.python.pylint.PylintSensor;
import org.sonar.plugins.python.xunit.PythonXUnitSensor;

public class PythonPlugin implements Plugin {

  private static final String PYTHON_CATEGORY = "Python";

  // Subcategories
  private static final String GENERAL = "General";
  private static final String TEST_AND_COVERAGE = "Tests and Coverage";
  private static final String PYLINT = "Pylint";
  private static final String DEPRECATED_PREFIX = "DEPRECATED : Use " + PythonCoverageSensor.REPORT_PATH_KEY + " instead. ";

  public static final String FILE_SUFFIXES_KEY = "sonar.python.file.suffixes";


  @Override
  public void define(Context context) {

    context.addExtensions(

      PropertyDefinition.builder(FILE_SUFFIXES_KEY)
        .index(10)
        .name("File Suffixes")
        .description("Comma-separated list of suffixes of Python files to analyze.")
        .category(PYTHON_CATEGORY)
        .subCategory(GENERAL)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("py")
        .build(),

      // COVERAGE
      PropertyDefinition.builder(PythonCoverageSensor.REPORT_PATH_KEY)
        .index(20)
        .name("Path to coverage report(s)")
        .description("Path to coverage reports. Ant patterns are accepted for relative path. The reports have to conform to the Cobertura XML format.")
        .category(PYTHON_CATEGORY)
        .subCategory(TEST_AND_COVERAGE)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue(PythonCoverageSensor.DEFAULT_REPORT_PATH)
        .build(),
      // deprecated
      PropertyDefinition.builder(PythonCoverageSensor.IT_REPORT_PATH_KEY)
        .index(21)
        .name("Path to coverage report(s) for integration tests")
        .description(DEPRECATED_PREFIX +
          "Path to coverage reports for integration tests. Ant patterns are accepted for relative path. The reports have to conform to the Cobertura XML format.")
        .category(PYTHON_CATEGORY)
        .subCategory(TEST_AND_COVERAGE)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue(PythonCoverageSensor.IT_DEFAULT_REPORT_PATH)
        .build(),
      // deprecated
      PropertyDefinition.builder(PythonCoverageSensor.OVERALL_REPORT_PATH_KEY)
        .index(22)
        .name("Path to overall (combined UT+IT) coverage report(s)")
        .description(DEPRECATED_PREFIX +
          "Path to a report containing overall test coverage data (i.e. test coverage gained by all tests of all kinds). " +
          "Ant patterns are accepted for relative path. The reports have to conform to the Cobertura XML format.")
        .category(PYTHON_CATEGORY)
        .subCategory(TEST_AND_COVERAGE)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue(PythonCoverageSensor.OVERALL_DEFAULT_REPORT_PATH)
        .build(),

      // XUNIT
      PropertyDefinition.builder(PythonXUnitSensor.SKIP_DETAILS)
        .index(23)
        .name("Skip the details when importing the Xunit reports")
        .description("When enabled the test execution statistics is provided only on project level. Use this mode when paths in report are not found. Disabled by default.")
        .category(PYTHON_CATEGORY)
        .subCategory(TEST_AND_COVERAGE)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("false")
        .type(PropertyType.BOOLEAN)
        .build(),
      PropertyDefinition.builder(PythonXUnitSensor.REPORT_PATH_KEY)
        .index(24)
        .name("Path to xunit report(s)")
        .description("Path to the report of test execution, relative to project's root. Ant patterns are accepted. The reports have to conform to the junitreport XML format.")
        .category(PYTHON_CATEGORY)
        .subCategory(TEST_AND_COVERAGE)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue(PythonXUnitSensor.DEFAULT_REPORT_PATH)
        .build(),

      // PYLINT
      PropertyDefinition.builder(PylintConfiguration.PYLINT_CONFIG_KEY)
        .index(30)
        .name("Pylint configuration")
        .description("Path to the pylint configuration file to use in pylint analysis. Set to empty to use the default.")
        .category(PYTHON_CATEGORY)
        .subCategory(PYLINT)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("")
        .build(),
      PropertyDefinition.builder(PylintConfiguration.PYLINT_KEY)
        .index(31)
        .name("Pylint executable")
        .description("Path to the pylint executable to use in pylint analysis. Set to empty to use the default one.")
        .category(PYTHON_CATEGORY)
        .subCategory(PYLINT)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("")
        .build(),
      PropertyDefinition.builder(PylintImportSensor.REPORT_PATH_KEY)
        .index(32)
        .name("Pylint's reports")
        .description("Path to Pylint's report file, relative to projects root")
        .category(PYTHON_CATEGORY)
        .subCategory(PYLINT)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("")
        .build(),

      Python.class,

      PythonProfile.class,

      PythonSquidSensor.class,
      PythonRuleRepository.class);

    if (context.getRuntime().getProduct() != SonarProduct.SONARLINT) {
      context.addExtensions(
        PylintConfiguration.class,
        PylintSensor.class,
        PylintImportSensor.class,
        PylintRuleRepository.class,
        PythonXUnitSensor.class);
    }
  }

}
