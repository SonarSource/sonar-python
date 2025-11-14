/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import java.util.function.Predicate;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S7506")
public class DictionaryStaticKeyCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DICT_COMPREHENSION, ctx -> {
      DictCompExpression dictCompExpression = (DictCompExpression) ctx.syntaxNode();
      Expression keyExpression = dictCompExpression.keyExpression();
      if (isLiteralDictKey(keyExpression) || isAssignedFromLiteralDictKey(keyExpression)) {
        ctx.addIssue(keyExpression, "Don't use a static key in a dictionary comprehension.");
      }
    });
  }

  private static boolean isLiteralDictKey(Expression expression) {
    if (expression instanceof StringLiteral stringLiteral) {
      return !isFString(stringLiteral);
    }
    return false;
  }

  private static boolean isAssignedFromLiteralDictKey(Expression expression) {
    return Expressions.ifNameGetSingleAssignedNonNameValue(expression)
      .map(TreeUtils.toInstanceOfMapper(StringLiteral.class))
      .filter(Predicate.not(DictionaryStaticKeyCheck::isFString))
      .isPresent();
  }

  private static boolean isFString(StringLiteral stringLiteral) {
    return stringLiteral.stringElements().stream().anyMatch(StringElement::isInterpolated);
  }
}
