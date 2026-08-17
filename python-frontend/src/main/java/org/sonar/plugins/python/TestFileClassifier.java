/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.Locale;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.api.config.Configuration;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;

import static org.sonarsource.analyzer.commons.appsec.TestFileClassifier.HEURISTIC_DISABLED_KEY;

/**
 * Heuristically identifies files that are likely to contain test code.
 */
public final class TestFileClassifier {

  private static final Set<String> TEST_MODULE_ROOTS = Set.of(
    "pytest", "unittest", "doctest", "hypothesis", "robot", "behave", "pytest_bdd", "nose", "nose2",
    "testtools", "mock", "pytest_mock", "fixtures", "factory", "factory_boy", "freezegun", "responses",
    "requests_mock", "respx", "httpretty", "moto", "vcr", "testcontainers");
  private static final Set<String> TEST_MODULE_PREFIXES = Set.of("django.test", "twisted.trial");
  private static final Set<String> TEST_DIRECTORY_NAMES = Set.of("test", "tests", "testing");

  private TestFileClassifier() {
  }

  /**
   * Checks whether test source classification is explicitly configured.
   * @param config analysis configuration
   * @return whether the heuristic must be disabled
   */
  public static boolean isTestSourceConfigured(Configuration config) {
    return isPropertyConfigured(config, "sonar.tests")
      || config.getBoolean("sonar.python.testFileHeuristic.disabled").orElse(false)
      || config.getBoolean(HEURISTIC_DISABLED_KEY).orElse(false);
  }

  /**
   * Determines whether a parsed Python file likely contains test code.
   * @param pythonFile analyzed Python file
   * @param tree parsed file tree
   * @return whether the file is likely test content
   */
  public static boolean looksLikeTestFile(PythonFile pythonFile, @Nullable FileInput tree) {
    String path = pythonFile.uri() == null ? null : pythonFile.uri().getPath();
    return looksLikeTestFile(path == null ? pythonFile.fileName() : path, tree);
  }

  /**
   * Determines whether a path and optional parsed tree identify test content.
   * @param filePath file path to inspect
   * @param tree parsed file tree
   * @return whether the file is likely test content
   */
  public static boolean looksLikeTestFile(String filePath, @Nullable FileInput tree) {
    if (looksLikeTestFileByPath(filePath)) {
      return true;
    }
    if (tree == null || tree.statements() == null) {
      return false;
    }
    return looksLikeTestFileByStatements(tree.statements());
  }

  /**
   * Determines whether a file path identifies test content.
   * @param filePath file path to inspect
   * @return whether the path indicates test content
   */
  static boolean looksLikeTestFileByPath(String filePath) {
    if (filePath.isEmpty()) {
      return false;
    }
    String[] components = filePath.replace('\\', '/').split("/");
    for (int i = 0; i < components.length - 1; i++) {
      if (TEST_DIRECTORY_NAMES.contains(components[i].toLowerCase(Locale.ROOT))) {
        return true;
      }
    }
    return components[components.length - 1].toLowerCase(Locale.ROOT).contains("test");
  }

  /**
   * Determines whether top-level statements identify test content.
   * @param statements top-level statements
   * @return whether the statements indicate test content
   */
  private static boolean looksLikeTestFileByStatements(StatementList statements) {
    int functionCount = 0;
    int testFunctionCount = 0;
    for (Statement statement : statements.statements()) {
      if (isTestFrameworkImport(statement) || isTestClass(statement)) {
        return true;
      }
      if (statement instanceof FunctionDef functionDef) {
        functionCount++;
        if (functionDef.name().name().toLowerCase(Locale.ROOT).startsWith("test")) {
          testFunctionCount++;
        }
      }
    }
    return testFunctionCount * 2 > functionCount;
  }

  /**
   * Determines whether a statement imports a testing framework or utility.
   * @param statement top-level statement to inspect
   * @return whether the statement imports test-specific code
   */
  private static boolean isTestFrameworkImport(Statement statement) {
    if (statement instanceof ImportName importName) {
      return importName.modules().stream().anyMatch(aliasedName -> isTestModule(aliasedName.dottedName()));
    }
    if (statement instanceof ImportFrom importFrom && importFrom.module() != null) {
      return isTestModule(importFrom.module());
    }
    return false;
  }

  /**
   * Determines whether a module belongs to a testing ecosystem.
   * @param module imported module name
   * @return whether the module is test-specific
   */
  private static boolean isTestModule(DottedName module) {
    var names = module.names();
    if (names.isEmpty()) {
      return false;
    }
    if (TEST_MODULE_ROOTS.contains(names.get(0).name())) {
      return true;
    }
    return names.size() > 1 && TEST_MODULE_PREFIXES.contains(names.get(0).name() + "." + names.get(1).name());
  }

  /**
   * Determines whether a statement declares a top-level test class.
   * @param statement top-level statement to inspect
   * @return whether the statement is a test class declaration
   */
  private static boolean isTestClass(Statement statement) {
    return statement instanceof ClassDef classDef && classDef.name().name().startsWith("Test");
  }

  /**
   * Determines whether a configuration property has a non-blank value.
   * @param config analysis configuration
   * @param key property key
   * @return whether the property is configured
   */
  private static boolean isPropertyConfigured(Configuration config, String key) {
    return config.get(key).filter(value -> !value.isBlank()).isPresent();
  }
}
