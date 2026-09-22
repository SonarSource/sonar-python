/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.utils;

import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

public final class PydanticUtils {

  private static final TypeMatcher IS_PYDANTIC_MODEL = TypeMatchers.isOrExtendsType("pydantic.BaseModel");
  private static final TypeMatcher IS_PYDANTIC_PRIVATE_ATTR = TypeMatchers.isType("pydantic.PrivateAttr");

  private PydanticUtils() {
  }

  public static boolean isPydanticModel(SubscriptionContext ctx, ClassDef classDef) {
    return isPydanticModel(ctx, classDef.name());
  }

  public static boolean isPydanticModel(SubscriptionContext ctx, Expression expression) {
    return IS_PYDANTIC_MODEL.isTrueFor(expression, ctx);
  }

  public static boolean isPrivateAttrCall(SubscriptionContext ctx, Expression expression) {
    return Expressions.removeParentheses(expression) instanceof CallExpression callExpression &&
      IS_PYDANTIC_PRIVATE_ATTR.isTrueFor(callExpression.callee(), ctx);
  }
}
