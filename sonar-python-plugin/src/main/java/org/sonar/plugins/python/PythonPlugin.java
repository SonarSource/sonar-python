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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.Plugin;
import org.sonar.api.PropertyType;
import org.sonar.api.SonarProduct;
import org.sonar.api.SonarRuntime;
import org.sonar.api.config.PropertyDefinition;
import org.sonar.api.resources.Qualifiers;
import org.sonar.plugins.python.api.SonarLintCache;
import org.sonar.plugins.python.bandit.BanditRulesDefinition;
import org.sonar.plugins.python.bandit.BanditSensor;
import org.sonar.plugins.python.coverage.PythonCoverageSensor;
import org.sonar.plugins.python.flake8.Flake8RulesDefinition;
import org.sonar.plugins.python.flake8.Flake8Sensor;
import org.sonar.plugins.python.indexer.SonarLintPythonIndexer;
import org.sonar.plugins.python.mypy.MypyRulesDefinition;
import org.sonar.plugins.python.mypy.MypySensor;
import org.sonar.plugins.python.pylint.PylintRulesDefinition;
import org.sonar.plugins.python.pylint.PylintSensor;
import org.sonar.plugins.python.ruff.RuffRulesDefinition;
import org.sonar.plugins.python.ruff.RuffSensor;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;
import org.sonar.plugins.python.xunit.PythonXUnitSensor;

import static org.sonar.plugins.python.api.PythonVersionUtils.PYTHON_VERSION_KEY;

public class PythonPlugin implements Plugin {

  private static final String PYTHON_CATEGORY = "Python";
  private static final Logger LOG = LoggerFactory.getLogger(PythonPlugin.class);

  // Subcategories
  private static final String GENERAL = "General";
  private static final String TEST_AND_COVERAGE = "Tests and Coverage";
  private static final String EXTERNAL_ANALYZERS_CATEGORY = "External Analyzers";
  private static final String DEPRECATED_PREFIX = "DEPRECATED : Use " + PythonCoverageSensor.REPORT_PATHS_KEY + " instead. ";

  public static final String PYTHON_FILE_SUFFIXES_KEY = "sonar.python.file.suffixes";
  public static final String IPYNB_FILE_SUFFIXES_KEY = "sonar.ipynb.file.suffixes";

  @Override
  public void define(Context context) {

    context.addExtensions(
      PropertyDefinition.builder(PYTHON_FILE_SUFFIXES_KEY)
        .index(10)
        .name("File Suffixes")
        .description("List of suffixes of Python files to analyze.")
        .multiValues(true)
        .category(PYTHON_CATEGORY)
        .subCategory(GENERAL)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("py")
        .build(),

      PropertyDefinition.builder(IPYNB_FILE_SUFFIXES_KEY)
        .index(11)
        .name("IPython File Suffixes")
        .description("List of suffixes of IPython Notebooks files to analyze.")
        .multiValues(true)
        .category(PYTHON_CATEGORY)
        .subCategory(GENERAL)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("ipynb")
        .build(),

      PropertyDefinition.builder(PYTHON_VERSION_KEY)
      .index(12)
      .name("Python versions")
      .description("Comma-separated list of Python versions this project is compatible with.")
      .multiValues(true)
      .category(PYTHON_CATEGORY)
      .subCategory(GENERAL)
      .onQualifiers(Qualifiers.PROJECT)
      .build(),

      Python.class,

      PythonProfile.class,

      PythonSensor.class,
      PythonRuleRepository.class,
      AnalysisWarningsWrapper.class,

      IPynb.class,
      IPynbProfile.class,
      IPynbSensor.class,
      IPynbRuleRepository.class);

    SonarRuntime sonarRuntime = context.getRuntime();
    if (sonarRuntime.getProduct() != SonarProduct.SONARLINT) {
      addCoberturaExtensions(context);
      addXUnitExtensions(context);
      addPylintExtensions(context);
      addBanditExtensions(context);
      addFlake8Extensions(context);
      addMypyExtensions(context);
      addRuffExtensions(context);
    }

    if (sonarRuntime.getProduct() == SonarProduct.SONARLINT) {
      SonarLintPluginAPIManager sonarLintPluginAPIManager = new SonarLintPluginAPIManager();
      sonarLintPluginAPIManager.addSonarlintPythonIndexer(context, new SonarLintPluginAPIVersion());
    }
  }

  private static void addCoberturaExtensions(Context context) {
    context.addExtensions(
      PropertyDefinition.builder(PythonCoverageSensor.REPORT_PATHS_KEY)
        .index(20)
        .name("Path to coverage report(s)")
        .description("List of paths pointing to coverage reports. Ant patterns are accepted for relative path. " +
          "The reports have to conform to the Cobertura XML format.")
        .category(PYTHON_CATEGORY)
        .subCategory(TEST_AND_COVERAGE)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue(PythonCoverageSensor.DEFAULT_REPORT_PATH)
        .multiValues(true)
        .build(),
      // deprecated
      PropertyDefinition.builder(PythonCoverageSensor.REPORT_PATH_KEY)
        .index(21)
        .name("Path to coverage report")
        .description(DEPRECATED_PREFIX +
          "Path to a coverage report. Ant patterns are accepted for relative path. The report has to conform to the Cobertura XML format.")
        .category(PYTHON_CATEGORY)
        .subCategory(TEST_AND_COVERAGE)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("")
        .build(),
      PythonCoverageSensor.class);
  }

  private static void addXUnitExtensions(Context context) {
    context.addExtensions(
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
      PythonXUnitSensor.class);
  }

  private static void addBanditExtensions(Context context) {
    context.addExtensions(BanditSensor.class,
      PropertyDefinition.builder(BanditSensor.REPORT_PATH_KEY)
        .name("Bandit Report Files")
        .description("Paths (absolute or relative) to json files with Bandit issues.")
        .category(EXTERNAL_ANALYZERS_CATEGORY)
        .subCategory(PYTHON_CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .multiValues(true)
        .build(),
      BanditRulesDefinition.class);
  }

  private static void addPylintExtensions(Context context) {
    context.addExtensions(PylintSensor.class,
      PropertyDefinition.builder(PylintSensor.REPORT_PATH_KEY)
        .name("Pylint Report Files")
        .description("Paths (absolute or relative) to report files with Pylint issues.")
        .category(EXTERNAL_ANALYZERS_CATEGORY)
        .subCategory(PYTHON_CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .multiValues(true)
        .build(),
      PylintRulesDefinition.class);
  }

  private static void addFlake8Extensions(Context context) {
    context.addExtensions(Flake8Sensor.class,
      PropertyDefinition.builder(Flake8Sensor.REPORT_PATH_KEY)
        .name("Flake8 Report Files")
        .description("Paths (absolute or relative) to report files with Flake8 issues.")
        .category(EXTERNAL_ANALYZERS_CATEGORY)
        .subCategory(PYTHON_CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .multiValues(true)
        .build(),
      Flake8RulesDefinition.class);
  }

  private static void addMypyExtensions(Context context) {
    context.addExtensions(MypySensor.class,
      PropertyDefinition.builder(MypySensor.REPORT_PATH_KEY)
        .name("Mypy Report Files")
        .description("Paths (absolute or relative) to report files with Mypy issues.")
        .category(EXTERNAL_ANALYZERS_CATEGORY)
        .subCategory(PYTHON_CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .multiValues(true)
        .build(),
      MypyRulesDefinition.class);
  }

  private static void addRuffExtensions(Context context) {
    context.addExtensions(RuffSensor.class,
      PropertyDefinition.builder(RuffSensor.REPORT_PATH_KEY)
        .name("Ruff Report Files")
        .description("Paths (absolute or relative) to report files with Ruff issues.")
        .category(EXTERNAL_ANALYZERS_CATEGORY)
        .subCategory(PYTHON_CATEGORY)
        .onQualifiers(Qualifiers.PROJECT)
        .multiValues(true)
        .build(),
      RuffRulesDefinition.class);
  }

  static class SonarLintPluginAPIManager {

    public void addSonarlintPythonIndexer(Context context, SonarLintPluginAPIVersion sonarLintPluginAPIVersion) {
      if (sonarLintPluginAPIVersion.isDependencyAvailable()) {
        // Only SonarLintPythonIndexer has the ModuleFileListener dependency.
        // However, SonarLintCache can only be used with SonarLintPythonIndexer present at the moment.
        // Hence, we also add it here, even if it technically does not share the dependency.
        //
        // Furthermore, with recent versions of SonarLint, the ModuleFileListener dependency should always be available.
        //
        // Attention:
        // The constructors of PythonSensor currently expect both, SonarLintCache and SonarLintPythonIndexer, to always be available at the
        // same time for injection.
        // Thus, some care is required when making changes to the addExtension calls here.
        context.addExtension(SonarLintCache.class);
        context.addExtension(SonarLintPythonIndexer.class);
      } else {
        LOG.debug("Error while trying to inject SonarLintPythonIndexer");
      }
    }
  }

  static class SonarLintPluginAPIVersion {

    boolean isDependencyAvailable() {
      try {
        Class.forName("org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileListener");
      } catch (ClassNotFoundException e) {
        return false;
      }
      return true;
    }
  }
}
