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

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;
import org.sonar.python.checks.cdk.CdkUtils.ResolvedKeyValuePair;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;
import static org.sonar.python.checks.cdk.CdkPredicate.isString;
import static org.sonar.python.checks.cdk.CdkPredicate.isStringLiteral;

@Rule(key = "S6302")
public class PrivilegePolicyCheck extends AbstractCdkResourceCheck {

  private static final String MESSAGE = "Make sure granting all privileges is safe here.";
  private static final String SECONDARY_MESSAGE = "Related effect";

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_iam.PolicyStatement", (ctx, call) -> {
      @Nullable ExpressionFlow effect = CdkUtils.getArgument(ctx, call, "effect").orElse(null);
      CdkUtils.getArgumentAsList(ctx, call, "actions").ifPresent(actions -> {
        if (effect == null || effect.hasExpression(isFqn("aws_cdk.aws_iam.Effect.ALLOW"))) {
          checkWildcardAction(ctx, actions, effect);
        }
      });
    });


    checkFqn("aws_cdk.aws_iam.PolicyStatement.from_json", (ctx, call) ->
      getPolicyFromJson(ctx, call).ifPresent(json -> checkPolicyStatement(ctx, json)));


    checkFqn("aws_cdk.aws_iam.PolicyDocument.from_json", (ctx, call) ->
      getPolicyFromJson(ctx, call).ifPresent(json -> {
        List<DictionaryLiteral> statements = CdkUtils.getDictionaryPair(ctx, json, "Statement")
          .map(ResolvedKeyValuePair::value)
          .flatMap(CdkUtils::getList)
          .map(list -> CdkUtils.getDictionaryInList(ctx, list))
          .orElse(Collections.emptyList());
        statements.forEach(statement -> checkPolicyStatement(ctx, statement));
      }));
  }


  private static void checkPolicyStatement(SubscriptionContext ctx, DictionaryLiteral statement) {
    List<ResolvedKeyValuePair> pairs = CdkUtils.resolveDictionary(ctx, statement);
    CdkUtils.getDictionaryPair(pairs, "Effect")
      .map(ResolvedKeyValuePair::value)
      .filter(effect -> effect.hasExpression(isString("Allow")))
      .ifPresent(effect ->
        CdkUtils.getDictionaryPair(pairs, "Action")
          .map(ResolvedKeyValuePair::value)
          .ifPresent(action -> checkActionFromJson(ctx, action, effect)));
  }

  private static void checkActionFromJson(SubscriptionContext ctx, ExpressionFlow action, ExpressionFlow effect) {
    // In the JSON representation a wildcard can be represented as a single string which is not part of a list
    if (action.hasExpression(isStringLiteral())) {
      action.addIssueIf(isString("*"), MESSAGE, secondaryLocation(effect));
    } else {
      CdkUtils.getList(action).ifPresent(actions -> checkWildcardAction(ctx, actions, effect));
    }
  }

  private static void checkWildcardAction(SubscriptionContext ctx, ListLiteral list, ExpressionFlow effect) {
    CdkUtils.getListElements(ctx, list)
      .stream()
      .filter(expr -> expr.hasExpression(isString("*")))
      .findFirst()
      .ifPresent(wildcard -> {
        PreciseIssue issue = ctx.addIssue(wildcard.getLast(), MESSAGE);
        if (effect != null) {
          issue.secondary(secondaryLocation(effect));
        }
      });
  }


  private static IssueLocation secondaryLocation(ExpressionFlow flow) {
    return IssueLocation.preciseLocation(flow.locations().getLast().parent(), SECONDARY_MESSAGE);
  }

  private static Optional<DictionaryLiteral> getPolicyFromJson(SubscriptionContext ctx, CallExpression call) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, "obj", call.arguments()))
      .map(argument -> ExpressionFlow.build(ctx, argument.expression()))
      .flatMap(CdkUtils::getDictionary);
  }
}
