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

import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.Expressions;

@Rule(key = "S6252")
public class S3BucketVersioningCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Make sure using unversioned S3 bucket is safe here. Omitting 'versioned=True' disables S3 bucket" +
    " versioning. Make sure it is safe here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::visitNode);
  }

  public void visitNode(SubscriptionContext ctx) {
    CallExpression node = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(node.calleeSymbol()).ifPresent(nodeSymbol -> {
      if ("aws_cdk.aws_s3.Bucket".equals(nodeSymbol.fullyQualifiedName())) {
        Optional<RegularArgument> version = getVersionArgument(node.arguments());
        if (version.isPresent()) {
          version.filter(a -> isExpressionFalse(a.expression())).ifPresent(v -> ctx.addIssue(v, MESSAGE));
        } else {
          ctx.addIssue(node.callee(), MESSAGE);
        }
      }
    });
  }

  private static Optional<RegularArgument> getVersionArgument(List<Argument> args) {
    return args.stream()
      .map(RegularArgument.class::cast)
      .filter(a -> a.keywordArgument() != null)
      .filter(a -> "versioned".equals(a.keywordArgument().name()))
      .findAny();
  }

  private static boolean isExpressionFalse(Expression expression) {
    if (expression.firstToken() != null && "False".equals(expression.firstToken().value())){
      return true;
    }
    if (expression.is(Tree.Kind.NAME)) {
      Expression singleAssignedValue = Expressions.singleAssignedValue(((Name) expression));
      if (singleAssignedValue == null) {
        return "False".equals(expression.firstToken().value());
      }
      return isExpressionFalse(singleAssignedValue);
    }
    return false;
  }
}
