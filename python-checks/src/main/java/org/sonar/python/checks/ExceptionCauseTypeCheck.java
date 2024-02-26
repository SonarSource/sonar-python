/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
import org.sonar.plugins.python.api.types.InferredType;

import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;
import static org.sonar.plugins.python.api.types.BuiltinTypes.STR;

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
    InferredType causeType = cause.type();
    if (causeType.canBeOrExtend("type")) {
      // SONARPY-1666: Here we should only exclude type objects that represent Exception types
      return;
    }
    // TODO remove the test against str once type inference knows the complete hierarchy of str
    if ((!causeType.canBeOrExtend(BASE_EXCEPTION) && !causeType.canOnlyBe(NONE_TYPE)) || causeType.canOnlyBe(STR)) {
      ctx.addIssue(cause, "Replace this expression with an exception or None");
    }
  }

}
