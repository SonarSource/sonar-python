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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S7504")
public class UnnecessaryListCastCheck extends PythonSubscriptionCheck {
  private TypeCheckBuilder isListCallCheck;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, this::initChecks);
    context.registerSyntaxNodeConsumer(Tree.Kind.FOR_STMT, this::checkForStatements);
    context.registerSyntaxNodeConsumer(Tree.Kind.COMP_FOR, this::checkComprehensions);
  }

  private void initChecks(SubscriptionContext ctx) {
    isListCallCheck = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("list");
  }

  private void checkForStatements(SubscriptionContext ctx) {
    ForStatement stmt = ((ForStatement) ctx.syntaxNode());
    checkListCastCheck(stmt.testExpressions(), ctx);
  }

  private void checkComprehensions(SubscriptionContext ctx) {
    ComprehensionFor comprehensionFor = ((ComprehensionFor) ctx.syntaxNode());
    checkListCastCheck(List.of(comprehensionFor.iterable()), ctx);
  }

  private void checkListCastCheck(List<Expression> expressions, SubscriptionContext ctx) {
    hasListCallOnIterable(expressions)
      .ifPresent(listCall -> ctx.addIssue(listCall.callee(), "Remove this unnecessary `list()` call on an already iterable object."));
  }

  private Optional<CallExpression> hasListCallOnIterable(List<Expression> testExpressions) {
    if (testExpressions.size() == 1
      && testExpressions.get(0) instanceof CallExpression callExpression
      && isListCall(callExpression)
      && hasOnlyOneRegularArg(callExpression)) {
      return Optional.of(callExpression);
    }
    return Optional.empty();
  }

  private boolean isListCall(CallExpression callExpression) {
    return isListCallCheck.check(callExpression.callee().typeV2()) == TriBool.TRUE;
  }

  private static boolean hasOnlyOneRegularArg(CallExpression callExpression) {
    return callExpression.arguments().size() == 1 && callExpression.arguments().get(0) instanceof RegularArgument;
  }
}
