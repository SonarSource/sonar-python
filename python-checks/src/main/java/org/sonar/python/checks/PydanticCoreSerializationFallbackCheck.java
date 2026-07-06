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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8966")
public class PydanticCoreSerializationFallbackCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add a \"fallback\" parameter to this pydantic-core serialization call.";
  private static final String FALLBACK_KEYWORD = "fallback";

  private static final TypeMatcher SERIALIZATION_CALL_MATCHER = TypeMatchers.any(
    TypeMatchers.isType("pydantic_core.to_json"),
    TypeMatchers.isType("pydantic_core.to_jsonable_python"));

  private static final TypeMatcher IS_PYDANTIC_BASE_MODEL_INSTANCE = TypeMatchers.isObjectInstanceOf("pydantic.BaseModel");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PydanticCoreSerializationFallbackCheck::checkCall);
  }

  private static void checkCall(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    if (!SERIALIZATION_CALL_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    var arguments = callExpression.arguments();
    if (arguments.isEmpty()) {
      return;
    }

    var firstArg = arguments.get(0);
    if (!(firstArg instanceof RegularArgument regularArgument)) {
      return;
    }

    if (isFirstArgumentPydanticBaseModelInstance(regularArgument, ctx)) {
      return;
    }

    if (TreeUtils.argumentByKeyword(FALLBACK_KEYWORD, callExpression.arguments()) != null) {
      return;
    }

    ctx.addIssue(callExpression.callee(), MESSAGE);
  }

  private static boolean isFirstArgumentPydanticBaseModelInstance(RegularArgument firstArg, SubscriptionContext ctx) {
    if (firstArg.keywordArgument() == null) {
      return !IS_PYDANTIC_BASE_MODEL_INSTANCE.evaluateFor(firstArg.expression(), ctx).isFalse();
    }
    return false;
  }
}
