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
package org.sonar.python.checks.tests;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.tree.TreeUtils;

import javax.annotation.Nullable;

@Rule(key = "S5976")
public class GroupSimilarTestsParameterizedCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Group these similar tests into a single parameterized test.";
  private static final int MIN_GROUP_SIZE = 3;
  private static final int MAX_PARAMETER_COUNT = 3;
  private static final TypeMatcher NOT_IMPLEMENTED_ERROR_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("builtins.NotImplementedError"),
    TypeMatchers.isObjectInstanceOf("builtins.NotImplementedError"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> checkFileInput(ctx, (FileInput) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.TESTS;
  }

  private static void checkFileInput(SubscriptionContext ctx, FileInput fileInput) {
    var statements = fileInput.statements();
    if (statements == null) {
      return;
    }
    checkStatementListRecursively(ctx, statements);
  }

  private static void checkStatementListRecursively(SubscriptionContext ctx, StatementList statementList) {
    checkStatementList(ctx, statementList);
    for (Statement statement : statementList.statements()) {
      if (statement instanceof ClassDef classDef) {
        checkStatementListRecursively(ctx, classDef.body());
      }
    }
  }

  private static void checkStatementList(SubscriptionContext ctx, StatementList statementList) {
    List<FunctionDef> testFunctions = statementList.statements().stream()
      .filter(FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .filter(functionDef -> isCandidateTestFunction(ctx, functionDef))
      .toList();

    Set<FunctionDef> alreadyReported = new HashSet<>();
    for (int i = 0; i < testFunctions.size(); i++) {
      FunctionDef anchor = testFunctions.get(i);
      if (alreadyReported.contains(anchor)) {
        continue;
      }

      List<FunctionDef> similarFunctions = new ArrayList<>();
      similarFunctions.add(anchor);
      for (int j = i + 1; j < testFunctions.size(); j++) {
        FunctionDef candidate = testFunctions.get(j);
        if (areSimilarTests(anchor, candidate)) {
          similarFunctions.add(candidate);
        }
      }

      if (similarFunctions.size() >= MIN_GROUP_SIZE) {
        var issue = ctx.addIssue(anchor.name(), MESSAGE);
        similarFunctions.stream().skip(1).forEach(similarFunction -> issue.secondary(similarFunction.name(), "Similar test."));
        alreadyReported.addAll(similarFunctions);
      }
    }
  }

  private static boolean isCandidateTestFunction(SubscriptionContext ctx, FunctionDef functionDef) {
    return UnittestUtils.isTestMethodName(functionDef.name().name())
      && functionDef.decorators().isEmpty()
      && !isPlaceholderTest(functionDef, ctx)
      && (UnittestUtils.isPytestStyleTestFunction(functionDef, ctx.pythonFile().fileName()) || UnittestUtils.isWithinUnittestTestCase(functionDef));
  }

  private static boolean isPlaceholderTest(FunctionDef functionDef, SubscriptionContext ctx) {
    List<Statement> statements = functionDef.body().statements();
    if (statements.size() != 1) {
      return false;
    }
    Statement statement = statements.get(0);
    return statement.is(Kind.PASS_STMT) || isNotImplementedErrorRaise(statement, ctx);
  }

  private static boolean isNotImplementedErrorRaise(Statement statement, SubscriptionContext ctx) {
    if (!(statement instanceof RaiseStatement raiseStatement) || raiseStatement.expressions().size() != 1) {
      return false;
    }
    return isNotImplementedError(raiseStatement.expressions().get(0), ctx);
  }

  private static boolean isNotImplementedError(Expression expression, SubscriptionContext ctx) {
    return NOT_IMPLEMENTED_ERROR_MATCHER.isTrueFor(expression, ctx);
  }

  private static boolean areSimilarTests(FunctionDef left, FunctionDef right) {
    if (!sameNamePattern(left, right)) {
      return false;
    }
    if (!sameTestKind(left, right)) {
      return false;
    }
    if (!CheckUtils.areEquivalent(left.parameters(), right.parameters())) {
      return false;
    }
    var differences = new DifferenceCounter();
    return differences.areSimilar(left.body(), right.body());
  }

  private static boolean sameTestKind(FunctionDef left, FunctionDef right) {
    boolean leftUnittest = UnittestUtils.isWithinUnittestTestCase(left);
    return leftUnittest == UnittestUtils.isWithinUnittestTestCase(right);
  }

  private static boolean sameNamePattern(FunctionDef left, FunctionDef right) {
    return normalizeName(left).equals(normalizeName(right));
  }

  private static String normalizeName(FunctionDef functionDef) {
    String name = functionDef.name().name();
    int suffixStart = name.length();
    while (suffixStart > 0 && Character.isDigit(name.charAt(suffixStart - 1))) {
      suffixStart--;
    }
    return name.substring(0, suffixStart);
  }

  private static class DifferenceCounter {
    private final Set<LiteralDifference> differences = new HashSet<>();

    private boolean areSimilar(@Nullable Tree left, @Nullable Tree right) {
      if (left == right) {
        return true;
      }
      if (left == null || right == null) {
        return false;
      }
      if (left.getKind() != right.getKind() || left.children().size() != right.children().size()) {
        return false;
      }
      if (left.children().isEmpty() && right.children().isEmpty()) {
        return compareLeaves(left, right);
      }
      for (int i = 0; i < left.children().size(); i++) {
        if (!areSimilar(left.children().get(i), right.children().get(i))) {
          return false;
        }
      }
      return true;
    }

    private boolean compareLeaves(Tree left, Tree right) {
      String leftValue = left.firstToken() == null ? null : left.firstToken().value();
      String rightValue = right.firstToken() == null ? null : right.firstToken().value();
      if ((leftValue == null && rightValue == null) || (leftValue != null && leftValue.equals(rightValue))) {
        return true;
      }
      if (!isParameterizableLiteral(left) || !isParameterizableLiteral(right)) {
        return false;
      }
      differences.add(new LiteralDifference(left.getKind(), leftValue, rightValue));
      return differences.size() <= MAX_PARAMETER_COUNT;
    }

    private static boolean isParameterizableLiteral(Tree tree) {
      Tree candidate = tree.parent() != null ? tree.parent() : tree;
      return candidate.is(Kind.NUMERIC_LITERAL, Kind.STRING_LITERAL, Kind.STRING_ELEMENT, Kind.NONE)
        || TreeUtils.isBooleanLiteral(candidate);
    }

    private record LiteralDifference(Kind kind, @Nullable String leftValue, @Nullable String rightValue) {
    }
  }
}
