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
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.Expressions;

public abstract class AbstractS3BucketCheck extends PythonSubscriptionCheck {

  private static final String S3_BUCKET_FQN = "aws_cdk.aws_s3.Bucket";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::visitNode);
  }

  private void visitNode(SubscriptionContext ctx) {
    CallExpression node = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(node.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(S3_BUCKET_FQN::equals)
      .ifPresent(s -> visitBucketConstructor(ctx, node));
  }

  abstract void visitBucketConstructor(SubscriptionContext ctx, CallExpression bucket);

  protected static Optional<ArgumentTrace> getArgument(SubscriptionContext ctx, CallExpression callExpression, String argumentName) {
    return callExpression.arguments().stream()
      .map(RegularArgument.class::cast)
      .filter(regularArgument -> regularArgument.keywordArgument() != null)
      .filter(regularArgument -> argumentName.equals(regularArgument.keywordArgument().name()))
      .map(regularArgument -> ArgumentTrace.build(ctx, regularArgument))
      .findAny();
  }

  static class ArgumentTrace {

    private static final String TAIL_MESSAGE = "Propagated setting.";

    private final SubscriptionContext ctx;
    private final List<Expression> trace;

    ArgumentTrace(SubscriptionContext ctx, List<Expression> trace) {
      this.ctx = ctx;
      this.trace = Collections.unmodifiableList(trace);
    }

    private static ArgumentTrace build(SubscriptionContext ctx, RegularArgument argument) {
      return new ArgumentTrace(ctx, buildTrace(argument.expression()));
    }
    private static List<Expression> buildTrace(Expression expression) {
      List<Expression> trace = new ArrayList<>();
      trace.add(expression);
      if (expression.is(Tree.Kind.NAME)) {
        Expression singleAssignedValue = Expressions.singleAssignedValue(((Name) expression));
        if (singleAssignedValue != null) {
          trace.addAll(buildTrace(singleAssignedValue));
        }
      }
      return trace;
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

    public boolean hasExpression(Predicate<Expression> predicate) {
      return trace.stream().anyMatch(predicate);
    }

    public List<Expression> trace() {
      return trace;
    }
  }

  protected static boolean isFalse(Expression expression) {
    return Optional.ofNullable(expression.firstToken()).map(Token::value).filter("False"::equals).isPresent();
  }
}
