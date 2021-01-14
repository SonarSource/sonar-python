/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.checks.hotspots;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree.Kind;

import static org.sonar.python.checks.Expressions.isFalsy;

public abstract class AbstractCookieFlagCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, ctx -> {
      AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();
      getSubscriptionToCookies(assignment.lhsExpressions())
        .forEach(sub -> {
          if (isSettingFlag(sub, flagName()) && isFalsy(assignment.assignedValue())) {
            ctx.addIssue(assignment, message());
          }
        });
    });

    context.registerSyntaxNodeConsumer(Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol calleeSymbol = callExpression.calleeSymbol();
      if (calleeSymbol != null && sensitiveArgumentByFQN().containsKey(calleeSymbol.fullyQualifiedName())) {
        if (callExpression.arguments().stream().anyMatch(argument -> argument.is(Kind.UNPACKING_EXPR))) {
          return;
        }
        RegularArgument flagArgument = getArgument(callExpression.arguments(), sensitiveArgumentByFQN().get(calleeSymbol.fullyQualifiedName()), flagName());
        if (flagArgument == null || isFalsy(flagArgument.expression())) {
          ctx.addIssue(callExpression.callee(), message());
        }
      }
    });
  }

  @CheckForNull
  private static RegularArgument getArgument(List<Argument> arguments, int nArg, String argumentName) {
    return arguments.stream()
      // in call site of this function, argument will always be of type RegularArgument because we check no argument is unpacking
      .map(RegularArgument.class::cast)
      .filter(argument -> {
        Name keywordArgument = argument.keywordArgument();
        return keywordArgument != null && keywordArgument.name().equals(argumentName);
      })
      .findFirst().orElseGet(() -> checkSensitivePositionalArgument(arguments, nArg));
  }

  private static RegularArgument checkSensitivePositionalArgument(List<Argument> arguments, int nArg) {
    for (int i = 0; i < arguments.size(); i++) {
      // in call site of this function, argument will always be of type RegularArgument because we check no argument is unpacking
      RegularArgument arg = (RegularArgument) arguments.get(i);
      if (i == nArg) {
        return arg;
      }
      if (arg.keywordArgument() != null) {
        // not positional anymore
        return null;
      }
    }
    return null;
  }

  private static Stream<SubscriptionExpression> getSubscriptionToCookies(List<ExpressionList> lhsExpressions) {
    return lhsExpressions.stream()
      .flatMap(expressionList -> expressionList.expressions().stream())
      .filter(lhs -> {
        if (lhs.is(Kind.SUBSCRIPTION)) {
          SubscriptionExpression sub = (SubscriptionExpression) lhs;
          return getObject(sub.object()).type().canOnlyBe("http.cookies.SimpleCookie");
        }
        return false;
      })
      .map(SubscriptionExpression.class::cast);
  }

  private static boolean isSettingFlag(SubscriptionExpression sub, String flagName) {
    List<ExpressionList> subscripts = getSubscripts(sub);
    if (subscripts.size() == 1) {
      return false;
    }
    return subscripts.stream()
      .skip(1)
      .anyMatch(s -> s.expressions().size() == 1 && isFlagNameStringLiteral(s.expressions().get(0), flagName));
  }

  private static List<ExpressionList> getSubscripts(SubscriptionExpression sub) {
    Deque<ExpressionList> subscripts = new ArrayDeque<>();
    subscripts.addFirst(sub.subscripts());
    Expression object = sub.object();
    while (object.is(Kind.SUBSCRIPTION)) {
      subscripts.addFirst(((SubscriptionExpression) object).subscripts());
      object = ((SubscriptionExpression) object).object();
    }
    return new ArrayList<>(subscripts);
  }

  private static boolean isFlagNameStringLiteral(Expression expression, String value) {
    return expression.is(Kind.STRING_LITERAL) && ((StringLiteral) expression).trimmedQuotesValue().equalsIgnoreCase(value);
  }

  private static Expression getObject(Expression object) {
    if (object.is(Kind.SUBSCRIPTION)) {
      return getObject(((SubscriptionExpression) object).object());
    }
    return object;
  }

  abstract String flagName();
  abstract String message();
  abstract Map<String, Integer> sensitiveArgumentByFQN();
}
