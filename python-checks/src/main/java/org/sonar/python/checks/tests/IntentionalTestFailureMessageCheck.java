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

import java.util.List;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S9077")
public class IntentionalTestFailureMessageCheck extends PythonSubscriptionCheck {

  private static final String ASSERT_MESSAGE = "Replace this assertion with pytest.fail(...) and provide a message.";
  private static final String FAIL_MESSAGE = "Add a message explaining why this test fails.";

  private static final TypeMatcher PYTEST_FAIL_MATCHER = TypeMatchers.isType("pytest.fail");

  private boolean pytestImported;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> pytestImported = importsPytest((FileInput) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> checkAssertStatement(ctx, (AssertStatement) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> checkCallExpression(ctx, (CallExpression) ctx.syntaxNode()));
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private void checkAssertStatement(SubscriptionContext ctx, AssertStatement assertStatement) {
    // Without a pytest import we cannot know whether the runner is pytest or unittest;
    // skip assert False / assert 0 to avoid FPs when pytest.fail is unavailable.
    if (!pytestImported) {
      return;
    }
    Expression condition = Expressions.removeParentheses(assertStatement.condition());
    if (isFalseOrZeroLiteral(condition)) {
      ctx.addIssue(assertStatement, ASSERT_MESSAGE);
    }
  }

  private static boolean isFalseOrZeroLiteral(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return "False".equals(((Name) expression).name());
    }
    if (expression.is(Tree.Kind.NUMERIC_LITERAL)) {
      try {
        return ((NumericLiteral) expression).valueAsLong() == 0;
      } catch (NumberFormatException nfe) {
        return false;
      }
    }
    return false;
  }

  private static void checkCallExpression(SubscriptionContext ctx, CallExpression callExpression) {
    if (!PYTEST_FAIL_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    RegularArgument messageArgument = messageArgument(callExpression);
    if (hasNoMessage(messageArgument)) {
      ctx.addIssue(callExpression, FAIL_MESSAGE);
    }
  }

  /**
   * Resolves the failure message argument. Prefer {@code reason=} or a positional argument,
   * and also accept the legacy {@code msg=} keyword used by older pytest versions.
   */
  @Nullable
  private static RegularArgument messageArgument(CallExpression callExpression) {
    RegularArgument reasonOrPositional = TreeUtils.nthArgumentOrKeyword(0, "reason", callExpression.arguments());
    if (reasonOrPositional != null) {
      return reasonOrPositional;
    }
    return TreeUtils.argumentByKeyword("msg", callExpression.arguments());
  }

  private static boolean hasNoMessage(@Nullable RegularArgument reasonArgument) {
    if (reasonArgument == null) {
      return true;
    }
    Expression reasonExpression = reasonArgument.expression();
    return reasonExpression.is(Tree.Kind.STRING_LITERAL) && ((StringLiteral) reasonExpression).trimmedQuotesValue().trim().isEmpty();
  }

  private static boolean importsPytest(FileInput fileInput) {
    PytestImportVisitor visitor = new PytestImportVisitor();
    fileInput.accept(visitor);
    return visitor.found;
  }

  private static final class PytestImportVisitor extends BaseTreeVisitor {
    private boolean found;

    @Override
    public void visitImportName(ImportName importName) {
      if (found) {
        return;
      }
      for (AliasedName module : importName.modules()) {
        if (isPytestModule(module.dottedName())) {
          found = true;
          return;
        }
      }
    }

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      if (found) {
        return;
      }
      if (isPytestModule(importFrom.module())) {
        found = true;
      }
    }

    private static boolean isPytestModule(@Nullable DottedName dottedName) {
      if (dottedName == null) {
        return false;
      }
      List<Name> names = dottedName.names();
      return !names.isEmpty() && "pytest".equals(names.get(0).name());
    }
  }
}
