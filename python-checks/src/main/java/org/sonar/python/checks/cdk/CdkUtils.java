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
package org.sonar.python.checks.cdk;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.Expressions;

public class CdkUtils {

  private CdkUtils() {
  }

  public static Optional<Integer> getInt(Expression expression) {
    try {
      return Optional.of((int)((NumericLiteral) expression).valueAsLong());
    } catch (ClassCastException e) {
      return Optional.empty();
    }
  }

  public static Optional<String> getString(Expression expression) {
    try {
      return Optional.of(((StringLiteral) expression).trimmedQuotesValue());
    } catch (ClassCastException e) {
      return Optional.empty();
    }
  }

  /**
   * Resolve a particular argument of a call or get an empty optional if the argument is not set.
   */
  protected static Optional<ExpressionTrace> getArgument(SubscriptionContext ctx, CallExpression callExpression, String argumentName) {
    return callExpression.arguments().stream()
      .map(RegularArgument.class::cast)
      .filter(regularArgument -> regularArgument.keywordArgument() != null)
      .filter(regularArgument -> argumentName.equals(regularArgument.keywordArgument().name()))
      .map(regularArgument -> ExpressionTrace.build(ctx, regularArgument.expression()))
      .findAny();
  }

  /**
   * Resolve the key and value of a dictionary element or get an empty optional if the element is an UnpackingExpression.
   */
  public static Optional<ResolvedKeyValuePair> getKeyValuePair(SubscriptionContext ctx, DictionaryLiteralElement element) {
    return element.is(Tree.Kind.KEY_VALUE_PAIR) ? Optional.of(ResolvedKeyValuePair.build(ctx, (KeyValuePair) element)) : Optional.empty();
  }

  /**
   * The expression trace reflects the complete flow of a value across the code.
   * It serves as a resolution path from the use of the expression up to the value assignment.
   * For example, if the value of an argument expression did not occur directly in the function call, the value can be traced back.
   * The trace allows on the one hand to check the assigned value
   * and on the other hand to display the assignment path of the relevant value when creating an issue.
   */
  static class ExpressionTrace {

    private static final String TAIL_MESSAGE = "Propagated setting.";

    private final SubscriptionContext ctx;
    private final List<Expression> trace;

    private ExpressionTrace(SubscriptionContext ctx, List<Expression> trace) {
      this.ctx = ctx;
      this.trace = Collections.unmodifiableList(trace);
    }

    protected static ExpressionTrace build(SubscriptionContext ctx, Expression expression) {
      List<Expression> trace = new ArrayList<>();
      buildTrace(expression, trace);
      return new ExpressionTrace(ctx, trace);
    }

    static void buildTrace(Expression expression, List<Expression> trace) {
      trace.add(expression);
      if (expression.is(Tree.Kind.NAME)) {
        Expression singleAssignedValue = Expressions.singleAssignedValue(((Name) expression));
        if (singleAssignedValue != null && !trace.contains(singleAssignedValue)) {
          buildTrace(singleAssignedValue, trace);
        }
      }
    }

    public void addIssue(String primaryMessage) {
      PythonCheck.PreciseIssue issue = ctx.addIssue(trace.get(0).parent(), primaryMessage);
      trace.stream().skip(1).forEach(expression -> issue.secondary(expression.parent(), TAIL_MESSAGE));
    }

    public void addIssueIf(Predicate<Expression> predicate, String primaryMessage) {
      if (hasExpression(predicate)) {
        addIssue(primaryMessage);
      }
    }

    public void addIssueIf(Predicate<Expression> predicate, String primaryMessage, CallExpression call) {
      if (hasExpression(predicate)) {
        ctx.addIssue(call.callee(), primaryMessage);
      }
    }

    public boolean hasExpression(Predicate<Expression> predicate) {
      return trace.stream().anyMatch(predicate);
    }

    public Optional<Expression> getExpression(Predicate<Expression> predicate) {
      return trace.stream().filter(predicate).findFirst();
    }

    public List<Expression> trace() {
      return trace;
    }
  }

  /**
   * Dataclass to store a resolved KeyValuePair structure
   */
  static class ResolvedKeyValuePair {

    final ExpressionTrace key;
    final ExpressionTrace value;

    private ResolvedKeyValuePair(ExpressionTrace key, ExpressionTrace value) {
      this.key = key;
      this.value = value;
    }

    static ResolvedKeyValuePair build(SubscriptionContext ctx, KeyValuePair pair) {
      return new ResolvedKeyValuePair(ExpressionTrace.build(ctx, pair.key()), ExpressionTrace.build(ctx, pair.value()));
    }
  }
}
