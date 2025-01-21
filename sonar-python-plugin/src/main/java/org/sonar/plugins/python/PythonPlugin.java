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

import org.sonar.api.Plugin;
import org.sonar.api.SonarProduct;
import org.sonar.api.SonarRuntime;
import org.sonar.api.config.PropertyDefinition;
import org.sonar.api.resources.Qualifiers;
import org.sonar.plugins.python.warnings.AnalysisWarningsWrapper;

import static org.sonar.plugins.python.api.PythonVersionUtils.PYTHON_VERSION_KEY;

public class PythonPlugin implements Plugin {


  @Override
  public void define(Context context) {

    context.addExtensions(
      PropertyDefinition.builder(PythonExtensions.PYTHON_FILE_SUFFIXES_KEY)
        .index(10)
        .name("File Suffixes")
        .description("List of suffixes of Python files to analyze.")
        .multiValues(true)
        .category(PythonExtensions.PYTHON_CATEGORY)
        .subCategory(PythonExtensions.GENERAL)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("py")
        .build(),

      PropertyDefinition.builder(PythonExtensions.IPYNB_FILE_SUFFIXES_KEY)
        .index(11)
        .name("IPython File Suffixes")
        .description("List of suffixes of IPython Notebooks files to analyze.")
        .multiValues(true)
        .category(PythonExtensions.PYTHON_CATEGORY)
        .subCategory(PythonExtensions.GENERAL)
        .onQualifiers(Qualifiers.PROJECT)
        .defaultValue("ipynb")
        .build(),

      PropertyDefinition.builder(PYTHON_VERSION_KEY)
      .index(12)
      .name("Python versions")
      .description("Comma-separated list of Python versions this project is compatible with.")
      .multiValues(true)
        .category(PythonExtensions.PYTHON_CATEGORY)
        .subCategory(PythonExtensions.GENERAL)
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
      PythonExtensions.addCoberturaExtensions(context);
      PythonExtensions.addXUnitExtensions(context);
      PythonExtensions.addPylintExtensions(context);
      PythonExtensions.addBanditExtensions(context);
      PythonExtensions.addFlake8Extensions(context);
      PythonExtensions.addMypyExtensions(context);
      PythonExtensions.addRuffExtensions(context);
    }

    if (sonarRuntime.getProduct() == SonarProduct.SONARLINT) {
      PythonExtensions.SonarLintPluginAPIManager sonarLintPluginAPIManager = new PythonExtensions.SonarLintPluginAPIManager();
      sonarLintPluginAPIManager.addSonarlintPythonIndexer(context, new PythonExtensions.SonarLintPluginAPIVersion());
    }
  }

}
