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
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7512")
public class InefficientDictIterationCheck extends PythonSubscriptionCheck {
  private TypeCheckBuilder dictItemsTypeCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, this::check);
  }

  private void initChecks(SubscriptionContext ctx) {
    dictItemsTypeCheck = ctx.typeChecker().typeCheckBuilder().isTypeWithName("dict.items");
  }

  private void check(SubscriptionContext ctx) {
    var forStatement = (ForStatement) ctx.syntaxNode();
    var hasIgnoredKey = hasIgnoredKey(forStatement);
    var hasIgnoredValue = hasIgnoredValue(forStatement);
    if (forStatement.testExpressions().size() == 1
        && (hasIgnoredKey || hasIgnoredValue)
        && (isSensitiveMethodCall(forStatement.testExpressions().get(0)) || isAssignedToSensitiveMethodCall(forStatement.testExpressions().get(0)))) {
      var message = hasIgnoredKey ? "Make this loop to iterate over the dict values." : "Make this loop to iterate directly over the dict.";
      ctx.addIssue(forStatement.testExpressions().get(0), message);
    }
  }

  private static boolean hasIgnoredKey(ForStatement forStatement) {
    return forStatement.expressions().size() == 2
           && forStatement.expressions().get(0) instanceof Name keyName
           && "_".equals(keyName.name());
  }

  private static boolean hasIgnoredValue(ForStatement forStatement) {
    return forStatement.expressions().size() == 2
           && forStatement.expressions().get(1) instanceof Name keyName
           && "_".equals(keyName.name());
  }


  private boolean isSensitiveMethodCall(@Nullable Expression expression) {
    return expression instanceof CallExpression callExpression
           && dictItemsTypeCheck.check(callExpression.callee().typeV2()) == TriBool.TRUE;
  }

  private boolean isAssignedToSensitiveMethodCall(Expression argumentExpression) {
    return argumentExpression instanceof Name name
           && getUsageCount(name) == 2
           && isSensitiveMethodCall(Expressions.singleAssignedValue(name));
  }

  private static int getUsageCount(Name name) {
    return Optional.ofNullable(name.symbolV2())
      .map(SymbolV2::usages)
      .map(List::size)
      .orElse(0);
  }
}
