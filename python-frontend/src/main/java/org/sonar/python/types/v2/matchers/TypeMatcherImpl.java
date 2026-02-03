/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.types.v2.matchers;

import org.sonar.api.Beta;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;

@Beta
public record TypeMatcherImpl(TypePredicate predicate) implements TypeMatcher {

  @Override
  public TriBool evaluateFor(Expression expr, SubscriptionContext ctx) {
    PythonType type = expr.typeV2();
    TypePredicateContext predicateContext = TypePredicateContext.of(ctx.typeTable());
    return TypePredicateUtils.evaluate(predicate, type, predicateContext);
  }

  @Override
  public boolean isTrueFor(Expression expr, SubscriptionContext ctx) {
    return evaluateFor(expr, ctx).isTrue();
  }

}