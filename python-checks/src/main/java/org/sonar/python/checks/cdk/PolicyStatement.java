/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.List;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.python.checks.cdk.CdkUtils.ExpressionFlow;

public abstract class PolicyStatement {

  enum Properties {
    PRINCIPALS("principals", "Principal"),
    CONDITIONS("conditions", "Condition"),
    EFFECT("effect", "Effect"),
    ACTIONS("actions", "Action"),
    RESOURCES("resources", "Resource");

    private final String constructorKey;
    private final String jsonKey;

    Properties(String constructorKey, String jsonKey) {
      this.constructorKey = constructorKey;
      this.jsonKey = jsonKey;
    }
  }

  @Nullable
  ExpressionFlow principals() {
    return getProperty(Properties.PRINCIPALS);
  }

  @Nullable
  ExpressionFlow conditions() {
    return getProperty(Properties.CONDITIONS);
  }

  @Nullable
  ExpressionFlow effect() {
    return getProperty(Properties.EFFECT);
  }

  @Nullable
  ExpressionFlow actions() {
    return getProperty(Properties.ACTIONS);
  }

  @Nullable
  ExpressionFlow resources() {
    return getProperty(Properties.RESOURCES);
  }

  @Nullable
  protected abstract ExpressionFlow getProperty(Properties property);

  public static PolicyStatement build(SubscriptionContext ctx, CallExpression call) {
    return new PolicyStatementFromConstructor(ctx, call);
  }

  public static PolicyStatement build(SubscriptionContext ctx, DictionaryLiteral json) {
    return new PolicyStatementFromJson(CdkUtils.resolveDictionary(ctx, json));
  }

  private static class PolicyStatementFromConstructor extends PolicyStatement {

    final SubscriptionContext ctx;
    final CallExpression call;

    public PolicyStatementFromConstructor(SubscriptionContext ctx, CallExpression call) {
      this.ctx = ctx;
      this.call = call;
    }

    @Override
    protected ExpressionFlow getProperty(Properties property) {
      return CdkUtils.getArgument(ctx, call, property.constructorKey).orElse(null);
    }
  }

  private static class PolicyStatementFromJson extends PolicyStatement {

    final List<CdkUtils.ResolvedKeyValuePair> pairs;

    private PolicyStatementFromJson(List<CdkUtils.ResolvedKeyValuePair> pairs) {
      this.pairs = pairs;
    }

    @Override
    protected ExpressionFlow getProperty(Properties property) {
      return CdkUtils.getDictionaryValue(pairs, property.jsonKey).orElse(null);
    }
  }
}
