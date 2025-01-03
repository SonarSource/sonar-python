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

import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

@Rule(key = "S5707")
public class ExceptionCauseTypeCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.RAISE_STMT, ctx -> {
      RaiseStatement raise = (RaiseStatement) ctx.syntaxNode();
      check(raise.fromExpression(), ctx);
    });
    context.registerSyntaxNodeConsumer(Kind.ASSIGNMENT_STMT, ctx -> {
      AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();
      Stream<Expression> lhsExpressions = assignment.lhsExpressions().stream()
        .flatMap(list -> list.expressions().stream());
      if (lhsExpressions.anyMatch(ExceptionCauseTypeCheck::isAccessToCause)) {
        check(assignment.assignedValue(), ctx);
      }
    });
  }

  private static boolean isAccessToCause(Expression e) {
    return e.is(Kind.QUALIFIED_EXPR) && ((QualifiedExpression) e).name().name().equals("__cause__");
  }

  private static void check(@Nullable Expression cause, SubscriptionContext ctx) {
    if (cause == null) {
      return;
    }
    PythonType causeType = cause.typeV2();
    TriBool inheritsFromBaseException = ctx.typeChecker().typeCheckBuilder().isInstanceOf("BaseException").check(causeType);
    TriBool isNoneType = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("NoneType").check(causeType);
    if (inheritsFromBaseException == TriBool.FALSE && isNoneType == TriBool.FALSE) {
      ctx.addIssue(cause, "Replace this expression with an exception or None");
    }
  }

}
