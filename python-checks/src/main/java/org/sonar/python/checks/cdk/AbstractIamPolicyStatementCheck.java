/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.cdk;

import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;

import static org.sonar.python.checks.cdk.CdkPredicate.isFqn;

public abstract class AbstractIamPolicyStatementCheck extends AbstractCdkResourceCheck {

  @Override
  protected void registerFqnConsumer() {
    checkFqn("aws_cdk.aws_iam.PolicyStatement", (ctx, call) ->
      checkPolicyStatement(PolicyStatement.build(ctx, call)));

    checkFqn("aws_cdk.aws_iam.PolicyStatement.from_json", (ctx, call) ->
      getDictionaryFromJson(ctx, call).ifPresent(json ->
        checkPolicyStatementFromJson(PolicyStatement.build(ctx, json))));

    checkFqn("aws_cdk.aws_iam.PolicyDocument.from_json", (ctx, call) ->
      getDictionaryFromJson(ctx, call).ifPresent(json -> getPolicyStatements(ctx, json)
        .forEach(statement -> checkPolicyStatementFromJson(PolicyStatement.build(ctx, statement)))));
  }

  protected void checkPolicyStatement(PolicyStatement policyStatement) {
    if (hasAllowEffect(policyStatement.effect())) {
      checkAllowingPolicyStatement(policyStatement);
    }
  }

  protected void checkPolicyStatementFromJson(PolicyStatement policyStatementFormJson) {
    checkPolicyStatement(policyStatementFormJson);
  }

  protected static boolean hasAllowEffect(@Nullable CdkUtils.ExpressionFlow effect) {
    // default is allow effect
    if (effect == null) {
      return true;
    }
    return effect.hasExpression(isFqn("aws_cdk.aws_iam.Effect.ALLOW").or(isJsonAllow()));
  }

  protected abstract void checkAllowingPolicyStatement(PolicyStatement policyStatement);

  protected static Optional<DictionaryLiteral> getDictionaryFromJson(SubscriptionContext ctx, CallExpression call) {
    return CdkUtils.getArgument(ctx, call, "obj", 0).flatMap(CdkUtils::getDictionary);
  }

  /**
   * Return a list of PolicyStatement json representation from a PolicyDocument.from_json call
   */
  protected static List<DictionaryLiteral> getPolicyStatements(SubscriptionContext ctx, DictionaryLiteral json) {
    return CdkUtils.getDictionaryPair(ctx, json, "Statement")
      .map(pair -> pair.value)
      .flatMap(CdkUtils::getList)
      .map(list -> CdkUtils.getDictionaryInList(ctx, list))
      .orElse(Collections.emptyList());
  }

  protected static CdkUtils.ExpressionFlow getSensitiveExpression(CdkUtils.ExpressionFlow expression, Predicate<Expression> predicate) {
    if (expression.hasExpression(predicate)) {
      return expression;
    } else {
      List<CdkUtils.ExpressionFlow> listElements = CdkUtils.getList(expression)
        .map(list -> CdkUtils.getListElements(expression.ctx(), list))
        .orElse(Collections.emptyList());

      return listElements.stream()
        .filter(expressionFlow -> expressionFlow.hasExpression(predicate))
        .findAny()
        .orElse(null);
    }
  }

  private static Predicate<Expression> isJsonAllow() {
    return expression -> CdkUtils.getString(expression).filter("allow"::equalsIgnoreCase).isPresent();
  }


}
