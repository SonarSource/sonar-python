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

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiConsumer;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.cdk.CdkUtils.getArgument;

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

  protected void checkFqn(String fqn, BiConsumer<SubscriptionContext, CallExpression> consumer) {
    fqnCallConsumers.put(fqn, consumer);
  }



}
