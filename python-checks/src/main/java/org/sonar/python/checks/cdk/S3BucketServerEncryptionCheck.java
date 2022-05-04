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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
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

  public static final String MESSAGE_OMITTING = "Omitting 'encryption' and 'encryption_key' disables server-side encryption. Make sure it is safe here.";
  public static final String MESSAGE_INCORRECT_TYPE = "Choose another compatible type of encryption";
  public static final String MESSAGE_SECONDARY = "Propagated settings.";
  private static final Set<String> AUTHORIZED_ENCRYPTION_TYPES = new HashSet<>(Arrays.asList("KMS", "S3_MANAGED", "KMS_MANAGED"));

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
        if (!isCorrectlyEncrypted(args)) {
          PreciseIssue issue = ctx.addIssue(node.callee(), MESSAGE_OMITTING);
          secondaryLocationExpression(args).stream()
            .skip(1)
            .forEach(arg -> issue.secondary(arg.parent(), MESSAGE_SECONDARY));
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

  private static boolean isCorrectlyEncrypted(List<RegularArgument> args) {
    Optional<RegularArgument> optEncryptionType = args.stream().filter(a -> "encryption".equals(getArgumentName(a))).findFirst();
    Optional<RegularArgument> optEncryptionKey = args.stream().filter(a -> "encryption_key".equals(getArgumentName(a))).findFirst();

    if (optEncryptionKey.isPresent()) {
      return !optEncryptionType.isPresent() || "KMS".equals(optEncryptionType.get().expression().lastToken().value());
    } else {
      if (optEncryptionType.isPresent()) {
        Expression expression = rootExpression(optEncryptionType.get().expression());
        Symbol symbol = ((QualifiedExpression) expression).symbol();
        if (symbol != null) {
          String fullName = symbol.fullyQualifiedName();
          if (fullName == null) {
            return false;
          }
          String[] split = fullName.split("\\.");
          String encryptionType = split[split.length - 1];
          boolean identifierIsCorrect = "aws_cdk.aws_s3.BucketEncryption".equals(fullName.split("." + encryptionType)[0]);
          return identifierIsCorrect && AUTHORIZED_ENCRYPTION_TYPES.contains(encryptionType);
        }
      }
      return false;
    }
  }

  private static Expression rootExpression(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      Expression singleAssignedValue = Expressions.singleAssignedValue(((Name) expression));
      if (singleAssignedValue != null) {
        return rootExpression(singleAssignedValue);
      }
    }
    return expression;
  }

  private static List<Tree> secondaryLocationExpression() {

    return Collections.emptyList();
  }
}
