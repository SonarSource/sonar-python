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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S7519")
public class ConstantValueDictComprehensionCheck extends PythonSubscriptionCheck {
  public static final String MESSAGE = "Replace with dict fromkeys method call";


  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.DICT_COMPREHENSION, ConstantValueDictComprehensionCheck::checkDictComprehension);
  }

  private static void checkDictComprehension(SubscriptionContext ctx) {
    var dictComprehension = (DictCompExpression) ctx.syntaxNode();
    if (isConstantValueDictComprehension(dictComprehension)) {
      ctx.addIssue(dictComprehension, MESSAGE);
    }
  }

  private static boolean isConstantValueDictComprehension(DictCompExpression dictComprehension) {
    if (!(dictComprehension.keyExpression() instanceof Name)
        || dictComprehension.comprehensionFor().nestedClause() != null) {
      return false;
    }

    if (dictComprehension.valueExpression() instanceof Name valueName) {
      var valueSymbol = valueName.symbolV2();
      return valueSymbol == null || !valueSymbol
        .usages()
        .stream()
        .map(UsageV2::tree)
        .allMatch(ut -> TreeUtils.firstAncestor(ut, dictComprehension::equals) != null);
    } else {
      return dictComprehension.valueExpression().is(Tree.Kind.NONE, Tree.Kind.STRING_LITERAL, Tree.Kind.NUMERIC_LITERAL, Tree.Kind.BOOLEAN_LITERAL_PATTERN);
    }
  }


}
