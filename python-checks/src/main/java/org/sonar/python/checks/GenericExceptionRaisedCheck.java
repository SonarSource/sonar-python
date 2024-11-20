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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.EXCEPTION;

@Rule(key = "S112")
public class GenericExceptionRaisedCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.RAISE_STMT, ctx -> {
      RaiseStatement raise = (RaiseStatement) ctx.syntaxNode();
      List<Expression> expressions = raise.expressions();
      if (expressions.isEmpty()) {
        return;
      }
      Expression expression = expressions.get(0);
      PythonType pythonType = expression.typeV2();
      TriBool isException = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName(EXCEPTION).check(pythonType);
      TriBool isBaseException = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName(BASE_EXCEPTION).check(pythonType);
      if (isException == TriBool.TRUE || isBaseException == TriBool.TRUE) {
        ctx.addIssue(expression, "Replace this generic exception class with a more specific one.");
      }
    });
  }
}
