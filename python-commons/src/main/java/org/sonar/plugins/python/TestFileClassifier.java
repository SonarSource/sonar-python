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

import java.util.Arrays;
import java.util.Locale;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.api.config.Configuration;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Statement;
import static org.sonarsource.analyzer.commons.appsec.TestFileClassifier.HEURISTIC_DISABLED_KEY;
import org.sonar.plugins.python.api.tree.StatementList;

/**
 * Heuristic classifier used when {@code sonar.tests} is not configured.
 * Determines whether a file should be treated as a test file for rule-execution
 * purposes, without affecting metric computation (which always uses the platform
 * {@code InputFile#type()}).
 */
public class TestFileClassifier {

  private static final Set<String> TEST_FRAMEWORK_MODULES = Set.of("unittest", "pytest");

  private TestFileClassifier() {
  }

  /**
   * Returns {@code true} when the project configuration explicitly controls which files are test
   * files, making the path-based heuristic unnecessary. Both {@link PythonScanner} and
   * {@link org.sonar.plugins.python.indexer.SonarQubePythonIndexer} use this to decide whether
   * to activate (and cache) the heuristic.
   */
  public static boolean isTestSourceConfigured(Configuration config) {
    return isPropertyConfigured(config, "sonar.tests")
      || config.getBoolean("sonar.python.testFileHeuristic.disabled").orElse(false)
      || config.getBoolean(HEURISTIC_DISABLED_KEY).orElse(false);
  }

  private static boolean isPropertyConfigured(Configuration config, String key) {
    return config.get(key).filter(v -> !v.isBlank()).isPresent();
  }

  /**
   * Path-only check — safe to call when no parsed tree is available (e.g. cache hits).
   * Matches files whose directory contains {@code test} or {@code tests}, or whose
   * filename follows the standard pytest discovery patterns ({@code test_*.py} / {@code *_test.py}).
   */
  static boolean looksLikeTestFileByPath(String filePath) {
    if (filePath.isEmpty()) {
      return false;
    }
    String normalizedPath = filePath.replace('\\', '/');
    String[] components = normalizedPath.split("/");

    // Check directory components (all but the last)
    boolean dirMatch = Arrays.stream(components, 0, components.length - 1)
      .map(c -> c.toLowerCase(Locale.ROOT))
      .anyMatch(dir -> "test".equals(dir) || "tests".equals(dir));
    if (dirMatch) {
      return true;
    }

    // Check filename
    String filename = components[components.length - 1].toLowerCase(Locale.ROOT);
    return filename.startsWith("test_") || filename.endsWith("_test.py");
  }

  /**
   * Full check — uses the parsed tree in addition to path heuristics.
   * Applies when a {@link FileInput} is available (i.e. during a full parse).
   */
  static boolean looksLikeTestFile(String filePath, @Nullable FileInput tree) {
    if (looksLikeTestFileByPath(filePath)) {
      return true;
    }
    if (tree == null) {
      return false;
    }
    return isImportBasedTestFile(tree) || isPytestPatternFile(tree);
  }

  private static boolean isImportBasedTestFile(FileInput tree) {
    StatementList statements = tree.statements();
    if (statements == null) {
      return false;
    }
    return statements.statements().stream().anyMatch(TestFileClassifier::isTestFrameworkImport);
  }

  private static boolean isTestFrameworkImport(Statement statement) {
    if (statement instanceof ImportName importName) {
      return importName.modules().stream()
        .map(aliasedName -> aliasedName.dottedName().names())
        .filter(names -> !names.isEmpty())
        .anyMatch(names -> TEST_FRAMEWORK_MODULES.contains(names.get(0).name()));
    }
    if (statement instanceof ImportFrom importFrom) {
      var module = importFrom.module();
      if (module != null) {
        var names = module.names();
        return !names.isEmpty() && TEST_FRAMEWORK_MODULES.contains(names.get(0).name());
      }
    }
    return false;
  }

  private static boolean isPytestPatternFile(FileInput tree) {
    StatementList statements = tree.statements();
    if (statements == null) {
      return false;
    }
    return statements.statements().stream()
      .filter(FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .anyMatch(f -> f.name().name().startsWith("test_") && containsAssert(f));
  }

  private static boolean containsAssert(FunctionDef functionDef) {
    var visitor = new AssertVisitor();
    functionDef.body().accept(visitor);
    return visitor.hasAssert;
  }

  private static class AssertVisitor extends BaseTreeVisitor {
    boolean hasAssert = false;

    @Override
    public void visitAssertStatement(AssertStatement assertStatement) {
      hasAssert = true;
    }
  }
}
