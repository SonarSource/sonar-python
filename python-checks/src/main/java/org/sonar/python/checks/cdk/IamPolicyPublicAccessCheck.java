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
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isFqnOf;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkUtils.getArgument;
import static org.sonar.python.checks.cdk.CdkUtils.getCall;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionary;
import static org.sonar.python.checks.cdk.CdkUtils.getDictionaryPair;
import static org.sonar.python.checks.cdk.CdkUtils.getList;
import static org.sonar.python.checks.cdk.CdkUtils.getListElements;
import static org.sonar.python.checks.cdk.CdkUtils.getListExpression;
import static org.sonar.python.checks.cdk.CdkUtils.resolveDictionary;

@Rule(key = "S6270")
public class IamPolicyPublicAccessCheck extends AbstractCdkResourceCheck {

  private static final String ISSUE_MESSAGE = "Make sure granting public access is safe here.";
  private static final String SECONDARY_MESSAGE = "Related effect.";

  private static final String IAM_EFFECT_ALLOW = "aws_cdk.aws_iam.Effect.ALLOW";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_iam.PolicyStatement", IamPolicyPublicAccessCheck::checkPolicyStatementConstructor);
    checkFqn("aws_cdk.aws_iam.PolicyStatement.from_json", IamPolicyPublicAccessCheck::checkPolicyStatementJson);
    checkFqn("aws_cdk.aws_iam.PolicyDocument.from_json", IamPolicyPublicAccessCheck::checkPolicyDocumentJson);
  }

  private static void checkPolicyStatementConstructor(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    // The rule should only trigger if "effect" is set to "ALLOW" or if it is unset
    Optional<CdkUtils.ExpressionFlow> effect = getArgument(subscriptionContext, callExpression, "effect");
    getArgument(subscriptionContext, callExpression, "principals")
      .flatMap(CdkUtils::getListExpression)
      .ifPresent(principals -> {
        Optional<Expression> allowSetting = effect.flatMap(expressionFlow -> expressionFlow.getExpression(isFqn(IAM_EFFECT_ALLOW)));
        if (effect.isEmpty() || allowSetting.isPresent()) {
          getListElements(subscriptionContext, principals)
            .forEach(principalElement -> raiseIssueIf(principalElement, isSensitivePrincipal(), allowSetting.orElse(null)));
        }
      });
  }

  private static void checkPolicyStatementJson(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    if (callExpression.arguments().isEmpty()) {
      return;
    }

    getArgument(subscriptionContext, callExpression, "obj", 0)
      .flatMap(CdkUtils::getDictionary)
      .ifPresent(dictionaryLiteral -> checkJsonEntry(subscriptionContext, dictionaryLiteral));
  }

  private static void checkPolicyDocumentJson(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    Optional<DictionaryLiteral> dictionaryLiteral = getArgument(subscriptionContext, callExpression, "obj", 0)
      .flatMap(CdkUtils::getDictionary);

    if (dictionaryLiteral.isEmpty()) {
      return;
    }

    Optional<ListLiteral> statement = CdkUtils.getDictionaryPair(subscriptionContext, dictionaryLiteral.get(), "Statement")
      .flatMap(resolvedKeyValuePair -> getList(resolvedKeyValuePair.value));

    if (statement.isEmpty()) {
      return;
    }

    getListElements(subscriptionContext, statement.get())
      .stream()
      .map(CdkUtils::getDictionary)
      .flatMap(Optional::stream)
      .forEach(innerDict -> checkJsonEntry(subscriptionContext, innerDict));
  }

  private static void checkJsonEntry(SubscriptionContext subscriptionContext, DictionaryLiteral dictionaryLiteral) {
    List<CdkUtils.ResolvedKeyValuePair> resolvedKeyValuePairs = resolveDictionary(subscriptionContext, dictionaryLiteral);

    Optional<CdkUtils.ResolvedKeyValuePair> effect = getDictionaryPair(resolvedKeyValuePairs, "Effect");
    Expression effectSetting = effect.flatMap(expressionFlow -> expressionFlow.value.getExpression(isString("Allow"))).orElse(null);

    if (effect.isPresent() && effectSetting == null) {
      // "Effect" is defined, and it is not equal to "Allow"
      return;
    }

    // Given an entry under the key "Principal", we want to raise an issue if the value is either
    //  1) string literal "*",
    //  2) dictionary in the format of { "AWS": "*" } or { "AWS": ["99999", ..., "*", ....] }
    getDictionaryPair(resolvedKeyValuePairs, "Principal").map(pair -> pair.value).ifPresent(principal -> {
      raiseIssueIf(principal, isString("*"), effectSetting);

      getDictionary(principal)
        .flatMap(innerDict -> getDictionaryPair(subscriptionContext, innerDict, "AWS"))
        .ifPresent(aws -> {
          // Check if the principal value is set to { "AWS": "*" }
          raiseIssueIf(aws.value, isString("*"), effectSetting);

          // Check if the principal value is set to { "AWS": ["99999", ..., "*", ....] }
          getListExpression(aws.value)
            .ifPresent(listLiteral ->
              getListElements(subscriptionContext, listLiteral)
                .forEach(principalElement -> raiseIssueIf(principalElement, isString("*"), effectSetting)));
        });
    });
  }

  private static Predicate<Expression> isSensitivePrincipal() {
    return isFqnOf(List.of("aws_cdk.aws_iam.StarPrincipal", "aws_cdk.aws_iam.AnyPrincipal"))
      .or(isSensitiveArnPrincipal());
  }

  private static Predicate<Expression> isSensitiveArnPrincipal() {
    return expression -> getCall(expression, "aws_cdk.aws_iam.ArnPrincipal")
      .map(callExpression -> TreeUtils.nthArgumentOrKeyword(0, "arn", callExpression.arguments()))
      .map(RegularArgument::expression)
      .filter(isString("*"))
      .isPresent();
  }

  private static void raiseIssueIf(CdkUtils.ExpressionFlow expressionFlow, Predicate<Expression> predicate, @Nullable Expression effect) {
    if (effect != null) {
      expressionFlow.addIssueIf(predicate, ISSUE_MESSAGE, IssueLocation.preciseLocation(effect, SECONDARY_MESSAGE));
    } else {
      expressionFlow.addIssueIf(predicate, ISSUE_MESSAGE);
    }
  }
}
