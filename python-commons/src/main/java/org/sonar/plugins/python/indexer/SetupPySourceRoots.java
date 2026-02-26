/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.plugins.python.indexer;

import com.google.common.annotations.VisibleForTesting;
import com.sonar.sslr.api.AstNode;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;
import org.sonar.python.tree.TreeUtils;

/**
 * Extracts source root directories from setup.py files.
 */
public class SetupPySourceRoots {

  private SetupPySourceRoots() {
  }

  /**
   * Extracts source root directories from setup.py content.
   *
   * @param setupPyContent the content of a setup.py file
   * @return list of source root paths (relative), empty if none found or on parse error
   */
  @VisibleForTesting
  static List<String> extract(String setupPyContent) {
    try {
      PythonParser parser = PythonParser.create();
      AstNode astNode = parser.parse(setupPyContent);
      PythonTreeMaker treeMaker = new PythonTreeMaker();
      FileInput fileInput = treeMaker.fileInput(astNode);

      SetupCallVisitor visitor = new SetupCallVisitor();
      fileInput.accept(visitor);

      return new ArrayList<>(visitor.sourceRoots);
    } catch (Exception e) {
      return List.of();
    }
  }

  /**
   * Extracts source root directories from a setup.py File.
   *
   * @param file the setup.py file
   * @return list of source root paths (relative), empty if none found or on parse error
   */
  public static List<String> extract(File file) {
    try {
      return extract(Files.readString(file.toPath()));
    } catch (IOException e) {
      return List.of();
    }
  }

  /**
   * Extracts source root directories from a setup.py File, preserving the config file location.
   *
   * <p>This method returns a {@link ConfigSourceRoots} that associates the extracted relative paths
   * with the config file, allowing callers to resolve absolute paths relative to the config file's
   * directory rather than the project base directory.
   *
   * @param file the setup.py file
   * @return ConfigSourceRoots containing the config file and its relative source roots
   */
  public static ConfigSourceRoots extractWithLocation(File file) {
    List<String> roots = extract(file);
    return new ConfigSourceRoots(file, roots);
  }

  private static class SetupCallVisitor extends BaseTreeVisitor {
    private final Set<String> sourceRoots = new LinkedHashSet<>();
    private final Map<String, Expression> variables = new HashMap<>();

    @Override
    public void visitAssignmentStatement(AssignmentStatement assignmentStatement) {
      ExpressionList lhs = assignmentStatement.lhsExpressions().get(0);
      if (lhs.expressions().size() == 1 && lhs.expressions().get(0) instanceof Name name) {
        variables.put(name.name(), assignmentStatement.assignedValue());
      }
      super.visitAssignmentStatement(assignmentStatement);
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Expression callee = callExpression.callee();

      if (callee instanceof Name name && "setup".equals(name.name())) {
        // First, check for dictionary unpacking arguments like setup(**config)
        extractFromUnpackingArguments(callExpression);

        // Then check for regular keyword arguments
        RegularArgument packagesArgument = TreeUtils.argumentByKeyword("packages", callExpression.arguments());
        if (packagesArgument != null) {
          extractFromPackages(packagesArgument.expression());
        }

        RegularArgument packageDirArgument = TreeUtils.argumentByKeyword("package_dir", callExpression.arguments());
        if (packageDirArgument != null) {
          extractFromPackageDir(packageDirArgument.expression());
        }
      }

      super.visitCallExpression(callExpression);
    }

    /**
     * Extracts source roots from dictionary unpacking arguments like setup(**config).
     * Handles patterns such as:
     * - setup(**{"package_dir": {"": "src"}})
     * - config = {"package_dir": {"": "src"}}; setup(**config)
     */
    private void extractFromUnpackingArguments(CallExpression callExpression) {
      for (Argument argument : callExpression.arguments()) {
        if (argument instanceof UnpackingExpression unpacking && TreeUtils.isDoubleStarExpression(unpacking)) {
          Expression unpackedExpr = resolveExpression(unpacking.expression());
          if (unpackedExpr instanceof DictionaryLiteral dictLiteral) {
            extractFromSetupConfigDict(dictLiteral);
          }
        }
      }
    }

    /**
     * Extracts source roots from a dictionary that represents setup() configuration.
     * Looks for "packages" and "package_dir" keys.
     */
    private void extractFromSetupConfigDict(DictionaryLiteral configDict) {
      for (var element : configDict.elements()) {
        if (element instanceof KeyValuePair keyValuePair) {
          String key = resolveToString(keyValuePair.key());
          if ("package_dir".equals(key)) {
            extractFromPackageDir(keyValuePair.value());
          } else if ("packages".equals(key)) {
            extractFromPackages(keyValuePair.value());
          }
        }
      }
    }

    private void extractFromPackages(Expression keyValuePair) {
      Expression valueExpr = resolveExpression(keyValuePair);
      if (valueExpr instanceof CallExpression call) {
        extractFromFindPackages(call);
      }
    }

    private void extractFromPackageDir(Expression keyValuePair) {
      Expression valueExpr = resolveExpression(keyValuePair);
      if (valueExpr instanceof DictionaryLiteral dictLiteral) {
        extractFromDictionary(dictLiteral);
      } else if (valueExpr instanceof CallExpression call) {
        extractFromFindPackages(call);
      }
    }

    private void extractFromDictionary(DictionaryLiteral dictLiteral) {
      for (var element : dictLiteral.elements()) {
        if (element instanceof KeyValuePair keyValuePair) {
          Expression value = keyValuePair.value();
          String resolvedValue = resolveToString(value);
          if (resolvedValue != null && !resolvedValue.isEmpty()) {
            sourceRoots.add(resolvedValue);
          }
        }
      }
    }

    private void extractFromFindPackages(CallExpression callExpression) {
      Expression callee = callExpression.callee();
      if (callee instanceof Name name && "find_packages".equals(name.name())) {
        RegularArgument whereArgument = TreeUtils.argumentByKeyword("where", callExpression.arguments());
        if(whereArgument!= null) {
          String whereValue = resolveToString(whereArgument.expression());
          if (whereValue != null && !whereValue.isEmpty()) {
            sourceRoots.add(whereValue);
          }
        }
      }
    }

    @Nullable
    private String resolveToString(Expression expression) {
      Expression resolved = resolveExpression(expression);
      if (resolved instanceof StringLiteral stringLiteral) {
        return stringLiteral.trimmedQuotesValue();
      }
      return null;
    }

    private Expression resolveExpression(Expression expression) {
      if (expression instanceof Name name) {
        Expression resolved = variables.get(name.name());
        if (resolved != null) {
          return resolveExpression(resolved);
        }
      }
      return expression;
    }
  }
}
