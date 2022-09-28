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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.Expressions;
import org.sonar.python.tree.TreeUtils;

public abstract class AbstractCdkResourceCheck extends PythonSubscriptionCheck {

  private final Map<String, BiConsumer<SubscriptionContext, CallExpression>> fqnCallConsumers = new HashMap<>();

  @Override
  public void initialize(SubscriptionCheck.Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::visitNode);
    registerFqnConsumer();
  }

  protected void visitNode(SubscriptionContext ctx) {
    CallExpression node = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(node.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .map(fqn -> fqnCallConsumers.getOrDefault(fqn, null))
      .ifPresent(consumer -> consumer.accept(ctx, node));
  }

  protected abstract void registerFqnConsumer();

  protected void checkFqn(String fqn, BiConsumer<SubscriptionContext, CallExpression> consumer) {
    fqnCallConsumers.put(fqn, consumer);
  }

  protected static Optional<ArgumentTrace> getArgument(SubscriptionContext ctx, CallExpression callExpression, String argumentName) {
    return callExpression.arguments().stream()
      .map(RegularArgument.class::cast)
      .filter(regularArgument -> regularArgument.keywordArgument() != null)
      .filter(regularArgument -> argumentName.equals(regularArgument.keywordArgument().name()))
      .map(regularArgument -> ArgumentTrace.build(ctx, regularArgument.expression()))
      .findAny();
  }

  /**
   * For compatibility with other classes and branches.
   * TODO Can be removed at the end of the sprint to reduce complexity.
   */
  public static class ArgumentTrace extends ExpressionTrace {
    private ArgumentTrace(SubscriptionContext ctx, List<Expression> trace) {
      super(ctx, trace);
    }

    protected static ArgumentTrace build(SubscriptionContext ctx, Expression expression) {
      List<Expression> trace = new ArrayList<>();
      buildTrace(expression, trace);
      return new ArgumentTrace(ctx, trace);
    }
  }

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
      PreciseIssue issue = ctx.addIssue(trace.get(0).parent(), primaryMessage);
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

  protected static Predicate<Expression> isFalse() {
    return expression -> Optional.ofNullable(expression.firstToken()).map(Token::value).filter("False"::equals).isPresent();
  }

  protected static Predicate<Expression> isNone() {
    return expression -> expression.is(Tree.Kind.NONE);
  }

  protected static Predicate<Expression> isFqn(String fqnValue) {
    return expression ->  Optional.ofNullable(TreeUtils.fullyQualifiedNameFromExpression(expression))
      .filter(fqnValue::equals)
      .isPresent();
  }

  protected static Optional<String> getStringValue(Expression expression) {
    try {
      return Optional.of(((StringLiteral) expression).trimmedQuotesValue());
    } catch (ClassCastException e) {
      return Optional.empty();
    }
  }

  /**
   * @return Predicate which tests if expression is a string and is equal the expected value
   */
  protected static Predicate<Expression> isStringValue(String expectedValue) {
    return expression -> getStringValue(expression).filter(expectedValue::equals).isPresent();
  }

  protected static Predicate<Expression> isSensitiveMethod(SubscriptionContext ctx, String methodFqn, String argName, Predicate<Expression> sensitiveValuePredicate) {
    return expression -> {
      if (!isFqn(methodFqn).test(expression)) {
        return false;
      }
      if (!expression.is(Tree.Kind.CALL_EXPR)) {
        return true;
      }

      Optional<AbstractCdkResourceCheck.ArgumentTrace> argTrace = getArgument(ctx, (CallExpression) expression, argName);
      if (argTrace.isEmpty()) {
        return true;
      }

      return argTrace.filter(trace -> trace.hasExpression(sensitiveValuePredicate)).isPresent();
    };
  }

}
