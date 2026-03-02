/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.EXCEPTION;

@Rule(key = "S112")
public class GenericExceptionRaisedCheck extends PythonSubscriptionCheck {

  private final TypeMatcher isExceptionOrBaseExceptionMatcher = TypeMatchers.any(
    TypeMatchers.isObjectOfType(EXCEPTION),
    TypeMatchers.isObjectOfType(BASE_EXCEPTION),
    TypeMatchers.isType(EXCEPTION),
    TypeMatchers.isType(BASE_EXCEPTION)
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.RAISE_STMT, ctx -> {
      RaiseStatement raise = (RaiseStatement) ctx.syntaxNode();
      List<Expression> expressions = raise.expressions();
      if (expressions.isEmpty()) {
        return;
      }

      Expression expression = expressions.get(0);
      if (!isExceptionOrBaseExceptionMatcher.isTrueFor(expression, ctx)) {
        return;
      }
      if (!isExceptionFunctionLocal(expression, raise)) {
        return;
      }

      ctx.addIssue(expression, "Replace this generic exception class with a more specific one.");
    });
  }

  private static boolean isExceptionFunctionLocal(Expression expression, RaiseStatement raise) {
    if (!(expression instanceof Name name)) return true;
    SymbolV2 symbolV2 = name.symbolV2();
    return symbolV2 == null || isLocalVariable(symbolV2, raise);
  }

  private static boolean isLocalVariable(SymbolV2 symbol, Tree raiseStatement) {
    Tree function = TreeUtils.firstAncestorOfKind(raiseStatement, Kind.FUNCDEF);
    if (function == null) {
      return false;
    }

    return symbol.getSingleBindingUsage()
      .filter(u -> !u.kind().equals(UsageV2.Kind.PARAMETER))
      .map(usage -> TreeUtils.firstAncestor(usage.tree(), t -> t == function) != null)
      .orElse(false);
  }
}
