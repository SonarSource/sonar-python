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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.DelStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.UnittestUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8997")
public class UseMonkeypatchFixtureCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use the \"monkeypatch\" fixture for temporary modifications instead of manually modifying global state.";
  private static final TypeMatcher ENVIRON_MATCHER = TypeMatchers.isType("os.environ");
  private static final TypeMatcher PYTEST_FIXTURE_MATCHER = TypeMatchers.withFQN("pytest.fixture");
  private static final TypeMatcher SYS_PATH_LIST_MUTATION_CALL =
    TypeMatchers.isFunctionOwnerSatisfying(TypeMatchers.isOrExtendsType("builtins.list"));
  private static final Set<String> PYTEST_LIFECYCLE_METHODS = Set.of(
    "setUp", "tearDown", "setUpClass", "tearDownClass",
    "setup_method", "teardown_method", "setup_class", "teardown_class",
    "setup_module", "teardown_module");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, UseMonkeypatchFixtureCheck::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.DEL_STMT, UseMonkeypatchFixtureCheck::checkDelete);
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, UseMonkeypatchFixtureCheck::checkCall);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkAssignment(SubscriptionContext ctx) {
    FunctionDef testFunction = enclosingPytestTestFunction(ctx, ctx.syntaxNode());
    if (testFunction == null) {
      return;
    }

    AssignmentStatement assignmentStatement = (AssignmentStatement) ctx.syntaxNode();
    for (ExpressionList lhsExpressionList : assignmentStatement.lhsExpressions()) {
      for (Expression lhsExpression : lhsExpressionList.expressions()) {
        checkManualModification(ctx, Expressions.removeParentheses(lhsExpression), testFunction);
      }
    }
  }

  private static void checkDelete(SubscriptionContext ctx) {
    FunctionDef testFunction = enclosingPytestTestFunction(ctx, ctx.syntaxNode());
    if (testFunction == null) {
      return;
    }

    DelStatement delStatement = (DelStatement) ctx.syntaxNode();
    for (Expression expression : delStatement.expressions()) {
      checkManualModification(ctx, Expressions.removeParentheses(expression), testFunction);
    }
  }

  private static void checkCall(SubscriptionContext ctx) {
    FunctionDef testFunction = enclosingPytestTestFunction(ctx, ctx.syntaxNode());
    if (testFunction == null) {
      return;
    }

    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (!(callExpression.callee() instanceof QualifiedExpression methodCall)) {
      return;
    }
    Expression receiver = Expressions.removeParentheses(methodCall.qualifier());
    if (!isSysPath(receiver, ctx) || !SYS_PATH_LIST_MUTATION_CALL.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }
    ctx.addIssue(callExpression, MESSAGE);
  }

  private static void checkManualModification(SubscriptionContext ctx, Expression expression, FunctionDef testFunction) {
    if (isOsEnvironSubscription(expression, ctx)
      || isSysPathSubscription(expression, ctx)
      || isManualModuleLevelAttributeAssignment(expression, testFunction)) {
      ctx.addIssue(expression, MESSAGE);
    }
  }

  private static boolean isOsEnvironSubscription(Expression expression, SubscriptionContext ctx) {
    return expression instanceof SubscriptionExpression subscriptionExpression
      && ENVIRON_MATCHER.isTrueFor(subscriptionExpression.object(), ctx);
  }

  private static boolean isSysPathSubscription(Expression expression, SubscriptionContext ctx) {
    return expression instanceof SubscriptionExpression subscriptionExpression
      && isSysPath(subscriptionExpression.object(), ctx);
  }

  private static boolean isSysPath(Expression expression, SubscriptionContext ctx) {
    return TreeUtils.fullyQualifiedNameFromExpression(expression)
      .filter("sys.path"::equals)
      .isPresent()
      || (expression instanceof QualifiedExpression qualifiedExpression
        && TypeMatchers.isType("sys").isTrueFor(qualifiedExpression.qualifier(), ctx)
        && "path".equals(qualifiedExpression.name().name()));
  }

  private static boolean isManualModuleLevelAttributeAssignment(Expression expression, FunctionDef testFunction) {
    if (!(expression instanceof QualifiedExpression qualifiedExpression)
      || CheckUtils.isSelf(qualifiedExpression.qualifier())) {
      return false;
    }
    Expression base = leftmostExpression(expression);
    if (base instanceof CallExpression) {
      // Mutating attributes of a call result (mocks, helpers, factories) is not global state.
      return false;
    }
    return !isLocallyBoundInTestFunction(base, testFunction);
  }

  private static Expression leftmostExpression(Expression expression) {
    Expression current = Expressions.removeParentheses(expression);
    while (true) {
      if (current instanceof QualifiedExpression qualifiedExpression) {
        current = Expressions.removeParentheses(qualifiedExpression.qualifier());
      } else if (current instanceof SubscriptionExpression subscriptionExpression) {
        current = Expressions.removeParentheses(subscriptionExpression.object());
      } else {
        return current;
      }
    }
  }

  private static boolean isLocallyBoundInTestFunction(Expression expression, FunctionDef testFunction) {
    if (!(expression instanceof Name name)) {
      return false;
    }
    SymbolV2 symbol = name.symbolV2();
    if (symbol == null) {
      return false;
    }
    for (UsageV2 usage : symbol.usages()) {
      if (!usage.isBindingUsage() || !isContainedIn(usage.tree(), testFunction)) {
        continue;
      }
      return true;
    }
    return false;
  }

  private static boolean isContainedIn(Tree tree, Tree ancestor) {
    for (Tree current = tree; current != null; current = current.parent()) {
      if (current == ancestor) {
        return true;
      }
    }
    return false;
  }

  private static FunctionDef enclosingPytestTestFunction(SubscriptionContext ctx, Tree tree) {
    FunctionDef functionDef = (FunctionDef) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF);
    if (functionDef == null) {
      return null;
    }
    if (isPytestFixture(functionDef, ctx) || isPytestLifecycleMethod(functionDef)) {
      return null;
    }
    if (!UnittestUtils.isPytestStyleTestFunction(functionDef, ctx.pythonFile().fileName())) {
      return null;
    }
    return functionDef;
  }

  private static boolean isPytestFixture(FunctionDef functionDef, SubscriptionContext ctx) {
    return functionDef.decorators().stream().anyMatch(decorator -> matchesPytestFixtureDecorator(decorator, ctx));
  }

  private static boolean matchesPytestFixtureDecorator(Decorator decorator, SubscriptionContext ctx) {
    Expression expression = decorator.expression();
    if (expression instanceof CallExpression callExpression) {
      return PYTEST_FIXTURE_MATCHER.isTrueFor(callExpression.callee(), ctx);
    }
    return PYTEST_FIXTURE_MATCHER.isTrueFor(expression, ctx);
  }

  private static boolean isPytestLifecycleMethod(FunctionDef functionDef) {
    return PYTEST_LIFECYCLE_METHODS.contains(functionDef.name().name());
  }
}
