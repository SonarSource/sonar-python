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
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.Expressions;

@Rule(key = "S6245")
public class S3BucketServerEncryptionCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Objects in the bucket are not encrypted. Make sure it is safe here.";
  public static final String MESSAGE_OMITTING = "Omitting 'encryption' disables server-side encryption. Make sure it is safe here.";
  public static final String MESSAGE_SECONDARY = "Propagated setting.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::visitNode);
  }

  public void visitNode(SubscriptionContext ctx) {
    CallExpression node = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(node.calleeSymbol())
      .filter(nodeSymbol -> "aws_cdk.aws_s3.Bucket".equals(nodeSymbol.fullyQualifiedName()))
      .ifPresent(nodeSymbol -> {
        List<RegularArgument> args = getEncryptionArguments(node.arguments());
        Optional<RegularArgument> optEncryptionType = args.stream().filter(a -> "encryption".equals(getArgumentName(a))).findFirst();
        Optional<RegularArgument> optEncryptionKey = args.stream().filter(a -> "encryption_key".equals(getArgumentName(a))).findFirst();
        if (optEncryptionType.isPresent()) {
          optEncryptionType.map(regArg -> propagatedSensitiveExpressions(regArg.expression()))
            .filter(trees -> !trees.isEmpty())
            .ifPresent(trees -> {
              PreciseIssue issue = ctx.addIssue(trees.get(0).parent(), MESSAGE);
              trees.stream().skip(1).forEach(expression -> issue.secondary(expression.parent(), MESSAGE_SECONDARY));
            });
        } else {
          if (!optEncryptionKey.isPresent()) {
            ctx.addIssue(node.callee(), MESSAGE_OMITTING);
          }
        }
      });
  }

  private static List<RegularArgument> getEncryptionArguments(List<Argument> args) {
    return args.stream()
      .map(RegularArgument.class::cast)
      .filter(a -> a.keywordArgument() != null)
      .filter(a -> "encryption".equals(getArgumentName(a)) || "encryption_key".equals(getArgumentName(a)))
      .collect(Collectors.toList());
  }

  private static String getArgumentName(RegularArgument argument) {
    Name keyword = argument.keywordArgument();
    if (keyword == null) {
      return "";
    }
    return keyword.name();
  }

  private static List<Tree> propagatedSensitiveExpressions(Expression expression) {
    List<Tree> trees = new ArrayList<>();
    if (expression.is(Tree.Kind.NAME)) {
      Expression singleAssignedValue = Expressions.singleAssignedValue(((Name) expression));
      if (singleAssignedValue != null) {
        trees.add(expression);
        List<Tree> subTrees = propagatedSensitiveExpressions(singleAssignedValue);
        if (subTrees.isEmpty()) {
          return Collections.emptyList();
        }
        trees.addAll(subTrees);
      }
    } else if (expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      if (isUnencrypted(expression)) {
        trees.add(expression);
      } else {
        return Collections.emptyList();
      }
    } else {
      trees.add(expression);
    }
    return trees;
  }

  private static boolean isUnencrypted(Expression expression) {
    Symbol symbol = ((QualifiedExpression) expression).symbol();
    if(symbol == null){
      return false;
    }
    return "aws_cdk.aws_s3.BucketEncryption.UNENCRYPTED".equals(symbol.fullyQualifiedName());
  }
}
