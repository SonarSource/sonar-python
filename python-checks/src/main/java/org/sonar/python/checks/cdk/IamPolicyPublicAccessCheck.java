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
import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;

import static org.sonar.python.checks.cdk.CdkPredicate.isDictionaryLiteral;
import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isFqnOf;
import static org.sonar.python.checks.cdk.CdkPredicate.isListLiteral;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getCall;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionary;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionaryPair;
import static org.sonar.python.checks.cdk.CdkUtils.getListExpression;

@Rule(key = "S6270")
public class IamPolicyPublicAccessCheck extends AbstractCdkResourceCheck {

  private static final String ISSUE_MESSAGE = "Make sure granting public access is safe here.";
  private static final String SECONDARY_MESSAGE = "Access is set to \"ALLOW\" here";

  private static final String IAM_EFFECT_ALLOW = "aws_cdk.aws_iam.Effect.ALLOW";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_iam.PolicyStatement", IamPolicyPublicAccessCheck::checkPolicyStatementConstructor);
    checkFqn("aws_cdk.aws_iam.PolicyStatement.from_json", IamPolicyPublicAccessCheck::checkPolicyStatementJson);
    checkFqn("aws_cdk.aws_iam.PolicyDocument.from_json", IamPolicyPublicAccessCheck::checkPolicyDocumentJson);
  }

  private static void checkPolicyStatementConstructor(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    // The rule should only trigger if "effect" is set to "ALLOW"
    Optional<Expression> sensitiveEffect = getSensitiveEffect(subscriptionContext, callExpression);
    if (sensitiveEffect.isEmpty()) {
      return;
    }

    getArgument(subscriptionContext, callExpression, "principals")
      .flatMap(CdkUtils::getListExpression)
      .ifPresent(listLiteral -> listLiteral.elements().expressions().stream()
        .filter(isSensitivePrincipal())
        .forEach(sensitivePrincipal -> raiseIssue(subscriptionContext, sensitivePrincipal, sensitiveEffect.get())));
  }

  private static void checkPolicyStatementJson(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    if (callExpression.arguments().isEmpty()) {
      return;
    }

    getArgument(subscriptionContext, callExpression, "obj", 0)
      .flatMap(e -> e.getExpression(e2 -> e2.is(Tree.Kind.DICTIONARY_LITERAL)))
      .map(DictionaryLiteral.class::cast)
      .ifPresent(dictionaryLiteral -> checkJsonEntry(subscriptionContext, dictionaryLiteral));
  }

  private static void checkPolicyDocumentJson(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    Optional<DictionaryLiteral> dictionaryLiteral = getArgument(subscriptionContext, callExpression, "obj", 0)
      .flatMap(e -> e.getExpression(isDictionaryLiteral()))
      .map(DictionaryLiteral.class::cast);

    if (dictionaryLiteral.isEmpty()) {
      return;
    }

    Optional<ListLiteral> statement = CdkUtils.getDictionaryPair(subscriptionContext, dictionaryLiteral.get(), "Statement")
      .flatMap(resolvedKeyValuePair -> resolvedKeyValuePair.value.getExpression(isListLiteral()))
      .map(ListLiteral.class::cast);

    if (statement.isEmpty()) {
      return;
    }

    for (Expression element : statement.get().elements().expressions()) {
      getDictionary(element).ifPresent(innerDict -> checkJsonEntry(subscriptionContext, innerDict));
    }
  }

  private static void checkJsonEntry(SubscriptionContext subscriptionContext, DictionaryLiteral dictionaryLiteral) {
    CdkUtils.DictionaryAsMap map = CdkUtils.DictionaryAsMap.build(subscriptionContext, dictionaryLiteral);

    Optional<Expression> sensitiveEffect = map.get("Effect", isString("Allow"));
    if (sensitiveEffect.isEmpty()) {
      return;
    }

    // Given an entry under the key "Principal", we want to raise an issue if the value is either
    //  1) string literal "*",
    //  2) dictionary in the format of { "AWS": "*" } or { "AWS": ["99999", ..., "*", ....] }
    map.getFlow("Principal").ifPresent(principal -> {
      principal.getExpression(isString("*"))
        .ifPresent(sensitiveStarPrincipal -> raiseIssue(subscriptionContext, sensitiveStarPrincipal, sensitiveEffect.get()));

      principal.getExpression(expression -> expression.is(Tree.Kind.DICTIONARY_LITERAL))
        .map(DictionaryLiteral.class::cast)
        .flatMap(innerDict -> getDictionaryPair(subscriptionContext, innerDict, "AWS"))
        .ifPresent(aws -> {
          // Check if the principal value is set to { "AWS": "*" }
          aws.value.addIssueIf(isString("*"), ISSUE_MESSAGE, IssueLocation.preciseLocation(sensitiveEffect.get(), SECONDARY_MESSAGE));

          // Check if the principal value is set to { "AWS": ["99999", ..., "*", ....] }
          getListExpression(aws.value).ifPresent(listLiteral -> listLiteral.elements().expressions().stream()
            .filter(isString("*"))
            .forEach(
              sensitivePrincipalElement -> raiseIssue(subscriptionContext, sensitivePrincipalElement, sensitiveEffect.get())));
        });
    });
  }

  private static Optional<Expression> getSensitiveEffect(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    return getArgument(subscriptionContext, callExpression, "effect")
      .map(CdkUtils.ExpressionFlow::getLast)
      .filter(isFqn(IAM_EFFECT_ALLOW));
  }

  private static Predicate<Expression> isSensitivePrincipal() {
    return isFqnOf(List.of("aws_cdk.aws_iam.StarPrincipal", "aws_cdk.aws_iam.AnyPrincipal"))
      .or(isSensitiveArnPrincipal());
  }

  private static Predicate<Expression> isSensitiveArnPrincipal() {
    return expression -> getCall(expression, "aws_cdk.aws_iam.ArnPrincipal")
      .filter(callExpression -> !callExpression.arguments().isEmpty())
      .map(callExpression -> callExpression.arguments().get(0))
      .filter(argument -> argument.is(Tree.Kind.REGULAR_ARGUMENT))
      .map(RegularArgument.class::cast)
      .map(RegularArgument::expression)
      .filter(isString("*"))
      .isPresent();
  }

  private static void raiseIssue(SubscriptionContext subscriptionContext, Tree primaryLocation, Tree secondaryLocation) {
    subscriptionContext.addIssue(primaryLocation, ISSUE_MESSAGE)
      .secondary(secondaryLocation, SECONDARY_MESSAGE);

  }
}
