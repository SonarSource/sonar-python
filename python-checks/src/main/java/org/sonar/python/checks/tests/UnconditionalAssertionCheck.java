/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.tests;

import java.util.List;
import java.util.Map;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;
import org.sonar.python.checks.CheckUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.NAME;
import static org.sonar.python.checks.CheckUtils.isConstant;

@Rule(key = "S5914")
public class UnconditionalAssertionCheck extends PythonSubscriptionCheck {

  private static final List<String> BOOLEAN_ASSERTIONS = List.of("assertTrue", "assertFalse");
  private static final List<String> NONE_ASSERTIONS = List.of("assertIsNone", "assertIsNotNone");

  private static final List<String> IS_ASSERTIONS = List.of("assertIs", "assertIsNot");
  private static final String BOOLEAN_MESSAGE = "Replace this expression; its boolean value is constant.";
  private static final String NONE_MESSAGE = "Remove this identity assertion; its value is constant.";

  private static final String IS_MESSAGE = "Replace this \"assertIs\" call with an \"assertEqual\" call.";
  private static final String IS_NOT_MESSAGE = "Replace this \"assertIsNot\" call with an \"assertNotEqual\" call.";

  private static final Map<List<String>, String> ASSERTION_MESSAGE_MAP = Map.of(
    BOOLEAN_ASSERTIONS, BOOLEAN_MESSAGE,
    NONE_ASSERTIONS, NONE_MESSAGE
  );

  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx ->
      reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(ctx.pythonFile()));

    context.registerSyntaxNodeConsumer(Tree.Kind.ASSERT_STMT, ctx -> {
      AssertStatement assertStatement = (AssertStatement) ctx.syntaxNode();
      Expression condition = assertStatement.condition();
      if (!condition.is(Tree.Kind.TUPLE) && CheckUtils.isConstant(condition)) {
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

      ASSERTION_MESSAGE_MAP.entrySet().stream()
        .filter(e -> e.getKey().contains(name))
        .findFirst()
        .ifPresent(e -> checkAssertion(ctx, TreeUtils.nthArgumentOrKeyword(0, "testValue", arguments), e.getValue()));

      if (IS_ASSERTIONS.contains(name)) {
        String message = "assertIs".equals(name) ? IS_MESSAGE : IS_NOT_MESSAGE;
        checkAssertion(ctx, TreeUtils.nthArgumentOrKeyword(0, "firstValue", arguments), message);
        checkAssertion(ctx, TreeUtils.nthArgumentOrKeyword(1, "secondValue", arguments), message);
      }
    });
  }


  private void checkAssertion(SubscriptionContext ctx, RegularArgument arg, String message) {
    if (isConstant(resolveArgument(arg))) {
      ctx.addIssue(arg, message);
    }
  }

  private Expression resolveArgument(RegularArgument argument) {
    Expression expression = argument.expression();
    if (expression.is(NAME)) {
      Set<Expression> valuesAtLocation = reachingDefinitionsAnalysis.valuesAtLocation(((Name) expression));
      if (valuesAtLocation.size() == 1) {
        return valuesAtLocation.iterator().next();
      }
    }
    return expression;
  }


  @Override
  public CheckScope scope() {
    return CheckScope.ALL;
  }
}
