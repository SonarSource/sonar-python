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

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiConsumer;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;

/**
 * Since most CDK related checks check arguments of method calls or object initializations,
 * this abstract class can be used to register CallExpression consumers for various fully qualified names.
 * For this purpose the method {@link #checkFqn(String, BiConsumer)} or {@link #checkFqns(Collection, BiConsumer)}
 * must be called in the {@link #registerFqnConsumer()} method which has to be implemented.
 */
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

  /**
   * Register a consumer for a single FQN
   */
  protected void checkFqn(String fqn, BiConsumer<SubscriptionContext, CallExpression> consumer) {
    fqnCallConsumers.put(fqn, consumer);
  }

  /**
   * Register a consumer for multiple FQNs
   */
  protected void checkFqns(Collection<String> suffixes, BiConsumer<SubscriptionContext, CallExpression> consumer) {
    suffixes.forEach(suffix -> checkFqn(suffix, consumer));
  }



}
