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

import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8998")
public class EmptyParametrizeValuesCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add at least one case to the parametrize values.";

  private static final TypeMatcher PYTEST_PARAMETRIZE_MATCHER = TypeMatchers.isType("pytest.mark.parametrize");
  private static final TypeMatcher POPULATING_METHOD_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("list.append"),
    TypeMatchers.isType("list.extend"),
    TypeMatchers.isType("list.insert"),
    TypeMatchers.isType("dict.update"),
    TypeMatchers.isType("dict.setdefault"),
    TypeMatchers.isType("set.add"),
    TypeMatchers.isType("set.update"),
    TypeMatchers.isType("typing.MutableMapping.update"),
    TypeMatchers.isType("typing.MutableMapping.setdefault"));
  private static final TypeMatcher EMPTY_NO_ARG_CONSTRUCTOR_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("list"),
    TypeMatchers.isType("dict"),
    TypeMatchers.isType("set"),
    TypeMatchers.isType("tuple"));
  private static final TypeMatcher RANGE_MATCHER = TypeMatchers.isType("range");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DECORATOR, EmptyParametrizeValuesCheck::checkDecorator);
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }

  private static void checkDecorator(SubscriptionContext ctx) {
    Decorator decorator = (Decorator) ctx.syntaxNode();
    Expression expression = decorator.expression();
    if (!expression.is(Tree.Kind.CALL_EXPR)) {
      return;
    }

    CallExpression callExpression = (CallExpression) expression;
    if (!PYTEST_PARAMETRIZE_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    RegularArgument valuesArgument = TreeUtils.nthArgumentOrKeyword(1, "argvalues", callExpression.arguments());
    if (valuesArgument == null) {
      return;
    }

    Expression valuesExpression = Expressions.removeParentheses(valuesArgument.expression());
    if (isEmptyParametrizeValues(valuesExpression, ctx) && !isCollectionPopulatedBeforeTest(valuesExpression, ctx)) {
      ctx.addIssue(valuesExpression, MESSAGE);
    }
  }

  private static boolean isEmptyParametrizeValues(Expression valuesExpression, SubscriptionContext ctx) {
    if (Expressions.isFalsy(valuesExpression) || isEmptyIterableConstructor(valuesExpression, ctx)) {
      return true;
    }
    if (!(valuesExpression instanceof Name name)) {
      return false;
    }
    Expression assigned = Expressions.singleAssignedValue(name);
    return assigned != null && isEmptyIterableConstructor(Expressions.removeParentheses(assigned), ctx);
  }

  private static boolean isEmptyIterableConstructor(Expression expression, SubscriptionContext ctx) {
    if (!(expression instanceof CallExpression callExpression)) {
      return false;
    }
    if (EMPTY_NO_ARG_CONSTRUCTOR_MATCHER.isTrueFor(callExpression.callee(), ctx) && callExpression.arguments().isEmpty()) {
      return true;
    }
    return isEmptyRange(callExpression, ctx);
  }

  private static boolean isEmptyRange(CallExpression callExpression, SubscriptionContext ctx) {
    if (!RANGE_MATCHER.isTrueFor(callExpression.callee(), ctx) || callExpression.arguments().size() != 1) {
      return false;
    }
    Argument argument = callExpression.arguments().get(0);
    if (!(argument instanceof RegularArgument regularArgument)) {
      return false;
    }
    Long bound = numericLiteralValue(Expressions.removeParentheses(regularArgument.expression()));
    return bound != null && bound <= 0L;
  }

  private static Long numericLiteralValue(Expression expression) {
    if (expression instanceof NumericLiteral numericLiteral) {
      try {
        return numericLiteral.valueAsLong();
      } catch (NumberFormatException e) {
        return null;
      }
    }
    if (expression instanceof UnaryExpression unaryExpression && "-".equals(unaryExpression.operator().value())) {
      Long positive = numericLiteralValue(Expressions.removeParentheses(unaryExpression.expression()));
      return positive == null ? null : -positive;
    }
    return null;
  }

  /**
   * Collections that start empty but are filled before tests run are not empty at runtime,
   * even when the mutation appears after the parametrize decorator in source order.
   * Only mutations that execute at import / class-definition time are considered.
   */
  private static boolean isCollectionPopulatedBeforeTest(Expression valuesExpression, SubscriptionContext ctx) {
    if (!(valuesExpression instanceof Name name)) {
      return false;
    }
    SymbolV2 symbol = name.symbolV2();
    if (symbol == null) {
      return false;
    }
    for (SymbolV2 relatedSymbol : symbolsIncludingSimpleAliases(symbol)) {
      for (UsageV2 usage : relatedSymbol.usages()) {
        Tree usageTree = usage.tree();
        if (isExecutedBeforeCollection(usageTree) && isPopulatingMethodCallOnReceiver(usageTree, ctx)) {
          return true;
        }
      }
    }
    return false;
  }

  private static Set<SymbolV2> symbolsIncludingSimpleAliases(SymbolV2 symbol) {
    Set<SymbolV2> symbols = new HashSet<>();
    symbols.add(symbol);
    for (UsageV2 usage : symbol.usages()) {
      symbols.addAll(collectSimpleAliasSymbols(usage.tree()));
    }
    return symbols;
  }

  private static Set<SymbolV2> collectSimpleAliasSymbols(Tree usageTree) {
    if (!(usageTree instanceof Name)) {
      return Set.of();
    }
    Tree assignmentTree = TreeUtils.firstAncestorOfKind(usageTree, Tree.Kind.ASSIGNMENT_STMT);
    if (!(assignmentTree instanceof AssignmentStatement assignment)) {
      return Set.of();
    }
    if (Expressions.removeParentheses(assignment.assignedValue()) != usageTree) {
      return Set.of();
    }
    Set<SymbolV2> aliases = new HashSet<>();
    for (ExpressionList lhsList : assignment.lhsExpressions()) {
      if (lhsList.expressions().size() != 1) {
        continue;
      }
      Expression lhs = lhsList.expressions().get(0);
      if (lhs instanceof Name lhsName && lhsName.symbolV2() != null) {
        aliases.add(lhsName.symbolV2());
      }
    }
    return aliases;
  }

  /**
   * Pytest collects parametrize values at import time. Mutations nested inside functions
   * (which are not executed during collection) must not suppress the issue.
   */
  private static boolean isExecutedBeforeCollection(Tree tree) {
    return TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF) == null;
  }

  private static boolean isPopulatingMethodCallOnReceiver(Tree usageTree, SubscriptionContext ctx) {
    Tree parent = usageTree.parent();
    if (!(parent instanceof QualifiedExpression qualifiedExpression) || qualifiedExpression.qualifier() != usageTree) {
      return false;
    }
    if (!(qualifiedExpression.parent() instanceof CallExpression callExpression)) {
      return false;
    }
    return POPULATING_METHOD_MATCHER.isTrueFor(callExpression.callee(), ctx);
  }
}
