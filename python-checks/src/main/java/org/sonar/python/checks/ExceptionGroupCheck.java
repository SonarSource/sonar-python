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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;

import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_GROUP_CLAUSE;

@Rule(key = "S6468")
public class ExceptionGroupCheck extends PythonSubscriptionCheck {
  private static final Set<String> EXCEPTION_GROUP = Set.of("ExceptionGroup", "BaseExceptionGroup");
  private static final String MESSAGE = "Avoid catching %s exception with 'except*'";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(EXCEPT_GROUP_CLAUSE, ctx -> {
      ExceptClause exceptClause = (ExceptClause) ctx.syntaxNode();
      Expression exception = exceptClause.exception();

      if (isExceptionGroup(exception)) {
        raiseIssue(ctx, exception);
      } else if (exception.is(Tree.Kind.TUPLE)) {
        Tuple exceptionTuple = (Tuple) exception;
        for (Expression exceptionEl : exceptionTuple.elements()) {
          if (isExceptionGroup(exceptionEl)) {
            raiseIssue(ctx, exceptionEl);
          }
        }
      }
    });
  }

  private static boolean isExceptionGroup(Expression exception) {
    // TODO : replace this logic by using type/symbol from typeshed
    return exception.is(Tree.Kind.NAME) && EXCEPTION_GROUP.contains(((Name) exception).name());
  }

  private static void raiseIssue(SubscriptionContext ctx, Expression exception) {
    ctx.addIssue(exception, String.format(MESSAGE, ((Name) exception).name()));
  }
}
