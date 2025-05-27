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

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S7500")
public class UnnecessaryComprehensionCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.GENERATOR_EXPR, UnnecessaryComprehensionCheck::checkComprehensionExpression);
    context.registerSyntaxNodeConsumer(Tree.Kind.LIST_COMPREHENSION, UnnecessaryComprehensionCheck::checkComprehensionExpression);
    context.registerSyntaxNodeConsumer(Tree.Kind.SET_COMPREHENSION, UnnecessaryComprehensionCheck::checkComprehensionExpression);
    context.registerSyntaxNodeConsumer(Tree.Kind.DICT_COMPREHENSION, UnnecessaryComprehensionCheck::checkDictComprehensionExpression);
  }

  private static void checkComprehensionExpression(SubscriptionContext ctx) {
    var comprehension = (ComprehensionExpression) ctx.syntaxNode();
    var valueExpression = comprehension.resultExpression();
    var loopExpression = Optional.of(comprehension)
      .map(ComprehensionExpression::comprehensionFor)
      .filter(comprehensionFor -> Objects.isNull(comprehensionFor.nestedClause()))
      .map(ComprehensionFor::loopExpression)
      .orElse(null);

    if (valueExpression instanceof Name valueName
        && loopExpression instanceof Name loopValueName
        && valueName.name().equals(loopValueName.name())
    ) {
      ctx.addIssue(comprehension, "Replace this comprehension with passing the iterable to the collection constructor call");
    }
  }

  private static void checkDictComprehensionExpression(SubscriptionContext ctx) {
    var comprehension = (DictCompExpression) ctx.syntaxNode();
    var keyExpression = comprehension.keyExpression();
    var valueExpression = comprehension.valueExpression();
    var loopExpressions = Optional.of(comprehension)
      .map(DictCompExpression::comprehensionFor)
      .filter(comprehensionFor -> Objects.isNull(comprehensionFor.nestedClause()))
      .map(ComprehensionFor::loopExpression)
      .map(TreeUtils.toInstanceOfMapper(Tuple.class))
      .map(Tuple::elements)
      .orElseGet(List::of);

    if (keyExpression instanceof Name keyName
        && valueExpression instanceof Name valueName
        && loopExpressions.size() == 2
        && loopExpressions.get(0) instanceof Name loopKeyName
        && loopExpressions.get(1) instanceof Name loopValueName
        && keyName.name().equals(loopKeyName.name())
        && valueName.name().equals(loopValueName.name())
    ) {
      ctx.addIssue(comprehension, "Replace this comprehension with passing the iterable to the dict constructor call");
    }
  }
}
