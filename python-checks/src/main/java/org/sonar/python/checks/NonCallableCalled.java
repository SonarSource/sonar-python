/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

import static org.sonar.python.tree.TreeUtils.nameFromExpression;

public abstract class NonCallableCalled extends PythonSubscriptionCheck {
  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      var callExpression = (CallExpression) ctx.syntaxNode();
      var callee = callExpression.callee();
      var calleeType = callee.typeV2();
      if (!isException(ctx, calleeType) && isCallMemberMissing(ctx, calleeType)) {
        String name = nameFromExpression(callee);
        var preciseIssue = ctx.addIssue(callee, message(calleeType, name));
        calleeType.definitionLocation()
          .ifPresent(location -> preciseIssue.secondary(location, "Definition."));
      }
    });
  }

  protected boolean isCallMemberMissing(SubscriptionContext ctx, PythonType calleeType) {
    return ctx.typeChecker().typeCheckBuilder().hasMember("__call__").check(calleeType) == TriBool.FALSE;
  }

  protected boolean isException(SubscriptionContext ctx, PythonType calleeType) {
    return !isExpectedTypeSource(ctx, calleeType);
  }

  protected abstract boolean isExpectedTypeSource(SubscriptionContext ctx, PythonType calleeType);

  protected static String addTypeName(PythonType type) {
    return type.displayName()
      .map(d -> " has type " + d + " and it")
      .orElse("");
  }

  protected String message(PythonType calleeType, @Nullable String name) {
    if (name != null) {
      return String.format("Fix this call; Previous type checks suggest that \"%s\"%s is not callable.", name, addTypeName(calleeType));
    }
    return String.format("Fix this call; Previous type checks suggest that this expression%s is not callable.", addTypeName(calleeType));
  }

}
