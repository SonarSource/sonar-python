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
package org.sonar.python.checks.tests;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.plugins.python.api.tree.Tree.Kind.NUMERIC_LITERAL;
import static org.sonar.plugins.python.api.tree.Tree.Kind.QUALIFIED_EXPR;
import static org.sonar.python.checks.utils.CheckUtils.isConstant;

@Rule(key = "S5914")
public class UnconditionalAssertionCheck extends PythonSubscriptionCheck {

  private static final Set<String> BOOLEAN_ASSERTIONS = Set.of("assertTrue", "assertFalse");
  private static final Set<String> NONE_ASSERTIONS = Set.of("assertIsNone", "assertIsNotNone");
  private static final Set<String> IS_ASSERTIONS = Set.of("assertIs", "assertIsNot");

  private static final String BOOLEAN_MESSAGE = "Replace this expression; its boolean value is constant.";
  private static final String NONE_MESSAGE = "Remove this identity assertion; its value is constant.";
  private static final String IS_MESSAGE = "Replace this \"assertIs\" call with an \"assertEqual\" call.";
  private static final String IS_NOT_MESSAGE = "Replace this \"assertIsNot\" call with an \"assertNotEqual\" call.";
  private static final String IS_SECONDARY_MESSAGE = "This expression creates a new object every time.";

  private static final Set<String> ACCEPTED_DECORATORS = Set.of("overload", "staticmethod", "classmethod");

  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx ->
      reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(ctx.pythonFile()));

    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> {
      AssertStatement assertStatement = (AssertStatement) ctx.syntaxNode();
      Expression condition = assertStatement.condition();
      if (!condition.is(Tree.Kind.TUPLE) && !isFalseOrZeroLiteral(condition) && CheckUtils.isConstant(condition)) {
        ctx.addIssue(condition, BOOLEAN_MESSAGE);
      }
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression call = (CallExpression) ctx.syntaxNode();
      Symbol symbol = call.calleeSymbol();
      if (symbol == null) {
        return;
      }

      String name = symbol.name();
      List<Argument> arguments = call.arguments();

      if (BOOLEAN_ASSERTIONS.contains(name)) {
        checkAssertion(arg -> checkBooleanAssertion(ctx, arg), 0, "expr", arguments);
      } else if (NONE_ASSERTIONS.contains(name)) {
        checkAssertion(arg -> checkNoneAssertion(ctx, call, arg),0, "expr", arguments);
      } else if (IS_ASSERTIONS.contains(name)) {
        String message = "assertIs".equals(name) ? IS_MESSAGE : IS_NOT_MESSAGE;
        checkAssertion(arg -> checkIsAssertion(ctx, call, arg, message), 0, "first", arguments);
        checkAssertion(arg -> checkIsAssertion(ctx, call, arg, message),1, "second", arguments);
      }
    });
  }

  private static void checkAssertion(Consumer<RegularArgument> checkConsumer, int argPosition, String keyword, List<Argument> arguments) {
    Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(argPosition, keyword, arguments)).ifPresent(checkConsumer);
  }

  /**
   * `assert False` or `assert 0` is often used to make a test fail.
   * Usually it is better to use another assertion or throw an AssertionException.
   * However, this rule is not intended to check this best practice.
   */
  private static boolean isFalseOrZeroLiteral(Expression expression) {
    if (expression.is(NAME)) {
      return "False".equals(((Name) expression).name());
    }
    if (expression.is(NUMERIC_LITERAL)) {
      return ((NumericLiteral) expression).valueAsLong() == 0;
    }
    return false;
  }

  private void checkNoneAssertion(SubscriptionContext ctx, CallExpression call, RegularArgument arg) {
    if (isUnconditional(arg)) {
      ctx.addIssue(call, NONE_MESSAGE);
    }
  }

  private void checkBooleanAssertion(SubscriptionContext ctx, RegularArgument arg) {
    if (isUnconditional(arg)) {
      ctx.addIssue(arg, BOOLEAN_MESSAGE);
    }
  }

  private static void checkIsAssertion(SubscriptionContext ctx, CallExpression call, RegularArgument arg, String message) {
    if (CheckUtils.isConstantCollectionLiteral(arg.expression())) {
      ctx.addIssue(call.callee(), message).secondary(arg, IS_SECONDARY_MESSAGE);
    }
  }

  private boolean isUnconditional(RegularArgument argument) {
    Expression expression = argument.expression();
    if (isConstant(expression)) {
      return true;
    }

    if (expression.is(NAME) || expression.is(QUALIFIED_EXPR)) {
      Symbol symbol = ((HasSymbol) expression).symbol();
      if (symbol != null) {
        if (isClassOrFunction(symbol)) {
          return true;
        }
        if (hasAnyNonlocalStatement(symbol)) {
          return false;
        }
      }
    }

    if (expression.is(NAME)) {
      Set<Expression> valuesAtLocation = reachingDefinitionsAnalysis.valuesAtLocation(((Name) expression));
      if (valuesAtLocation.size() == 1) {
        return CheckUtils.isImmutableConstant(valuesAtLocation.iterator().next());
      }
    }

    return false;
  }

  private static boolean hasAnyNonlocalStatement(Symbol symbol) {
    return symbol.usages().stream()
      .anyMatch(usage -> usage.tree().parent().is(Tree.Kind.NONLOCAL_STMT));
  }

  private static boolean isClassOrFunction(Symbol symbol) {
    if (symbol.is(Symbol.Kind.CLASS)) {
      return true;
    }
    if (symbol.is(Symbol.Kind.FUNCTION)) {
      // Avoid potential FPs with properties: only report on limited selection of "safe" decorators
      return ACCEPTED_DECORATORS.containsAll(((FunctionSymbol) symbol).decorators());
    }
    return false;
  }

  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
