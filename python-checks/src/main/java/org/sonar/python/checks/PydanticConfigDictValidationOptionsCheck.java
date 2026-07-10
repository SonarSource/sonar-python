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
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8953")
public class PydanticConfigDictValidationOptionsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Enable at least one of \"validate_by_alias\" or \"validate_by_name\".";
  private static final String SECONDARY_MESSAGE = "Also set to \"False\" here.";

  private static final String VALIDATE_BY_ALIAS = "validate_by_alias";
  private static final String VALIDATE_BY_NAME = "validate_by_name";

  private static final TypeMatcher IS_PYDANTIC_CONFIG_DICT = TypeMatchers.isType("pydantic.ConfigDict");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PydanticConfigDictValidationOptionsCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    if (!IS_PYDANTIC_CONFIG_DICT.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    RegularArgument validateByAliasArg = TreeUtils.argumentByKeyword(VALIDATE_BY_ALIAS, callExpression.arguments());
    RegularArgument validateByNameArg = TreeUtils.argumentByKeyword(VALIDATE_BY_NAME, callExpression.arguments());

    if (validateByAliasArg == null || validateByNameArg == null) {
      return;
    }

    if (Expressions.isFalsy(validateByAliasArg.expression()) && Expressions.isFalsy(validateByNameArg.expression())) {
      ctx.addIssue(validateByNameArg, MESSAGE).secondary(validateByAliasArg, SECONDARY_MESSAGE);
    }
  }

}
