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

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.config.Configuration;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonarsource.analyzer.commons.appsec.TestFileClassifier;

/**
 * Adds Python-specific structural signals to the shared test-file heuristic.
 */
public final class PythonTestFileClassifier {

  private static final String[] TEST_FILE_PATH_PATTERNS = {
    "**/test/**", "**/tests/**", "**/testing/**",
    "**/TEST/**", "**/TESTS/**", "**/TESTING/**",
    "**/*test*", "*test*", "**/*Test*", "*Test*", "**/*TEST*", "*TEST*"
  };
  private static final Set<String> TEST_MODULE_ROOTS = Set.of(
    "pytest", "unittest", "doctest", "hypothesis", "robot", "behave", "pytest_bdd", "nose", "nose2",
    "testtools", "pytest_mock", "factory_boy", "freezegun",
    "requests_mock", "respx", "httpretty", "moto", "testcontainers");
  private static final Set<String> TEST_MODULE_PREFIXES = Set.of("django.test", "twisted.trial");
  private final TestFileClassifier delegate;

  public PythonTestFileClassifier(Configuration configuration) {
    delegate = TestFileClassifier.of(configuration, PythonTestFileClassifier::hasTestContentContext, TEST_FILE_PATH_PATTERNS);
  }

  /**
   * Determines whether a Python file is likely to contain test code.
   * @param inputFile analyzed input file
   * @param tree parsed file tree, when parsing succeeded
   * @return whether the file is likely test content
   */
  public boolean looksLikeTestFile(InputFile inputFile, @Nullable FileInput tree) {
    return delegate.looksLikeTestFile(inputFile, new PythonContext(tree));
  }

  /**
   * Determines whether a parsed tree has Python-specific test signals.
   * @param tree parsed file tree
   * @return whether the tree contains test-specific imports or declarations
   */
  public static boolean hasTestContent(@Nullable FileInput tree) {
    return looksLikeTestFileByStatements(tree == null ? null : tree.statements());
  }

  private static boolean hasTestContentContext(TestFileClassifier.Context context) {
    return context instanceof PythonContext pythonContext
      && hasTestContent(pythonContext.tree);
  }

  private static final class PythonContext implements TestFileClassifier.Context {
    private final FileInput tree;

    private PythonContext(@Nullable FileInput tree) {
      this.tree = tree;
    }
  }

  /**
   * Determines whether top-level statements identify test content.
   * @param statements top-level statements
   * @return whether the statements indicate test content
   */
  private static boolean looksLikeTestFileByStatements(@Nullable StatementList statements) {
    if (statements == null) {
      return false;
    }
    int functionCount = 0;
    int testFunctionCount = 0;
    for (Statement statement : statements.statements()) {
      if (isTestFrameworkImport(statement) || isTestClass(statement)) {
        return true;
      }
      if (statement instanceof FunctionDef functionDef) {
        functionCount++;
        if (isTestShaped(functionDef.name().name())) {
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
      return importName.modules().stream().anyMatch(aliasedName -> isTestModule(moduleNames(aliasedName.dottedName())));
    }
    if (statement instanceof ImportFrom importFrom && importFrom.module() != null) {
      List<String> base = moduleNames(importFrom.module());
      if (isTestModule(base)) {
        return true;
      }
      return importFrom.importedNames().stream()
        .anyMatch(importedName -> isTestModule(concat(base, moduleNames(importedName.dottedName()))));
    }
    return false;
  }

  /**
   * Determines whether a dotted module name belongs to a testing ecosystem.
   * @param names segments of the imported module name
   * @return whether the module is test-specific
   */
  private static boolean isTestModule(List<String> names) {
    if (names.isEmpty()) {
      return false;
    }
    if (TEST_MODULE_ROOTS.contains(names.get(0))) {
      return true;
    }
    return names.size() > 1 && TEST_MODULE_PREFIXES.contains(names.get(0) + "." + names.get(1));
  }

  private static List<String> moduleNames(@Nullable DottedName dottedName) {
    if (dottedName == null) {
      return List.of();
    }
    return dottedName.names().stream().map(Name::name).toList();
  }

  private static List<String> concat(List<String> base, List<String> tail) {
    List<String> combined = new ArrayList<>(base);
    combined.addAll(tail);
    return combined;
  }

  private static boolean isTestShaped(String name) {
    String lowerCaseName = name.toLowerCase(Locale.ROOT);
    return lowerCaseName.startsWith("test")
      && (name.length() == 4 || !Character.isLowerCase(name.charAt(4)));
  }

  /**
   * Determines whether a statement declares a top-level test class.
   * @param statement top-level statement to inspect
   * @return whether the statement is a test class declaration
   */
  private static boolean isTestClass(Statement statement) {
    return statement instanceof ClassDef classDef
      && isTestShaped(classDef.name().name());
  }
}
