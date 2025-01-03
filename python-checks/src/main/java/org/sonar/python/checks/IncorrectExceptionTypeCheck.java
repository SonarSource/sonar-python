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

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.BuiltinSymbols;

import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;

@Rule(key = "S5632")
public class IncorrectExceptionTypeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Change this code so that it raises an object deriving from BaseException.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.RAISE_STMT, ctx -> {
      RaiseStatement raiseStatement = (RaiseStatement) ctx.syntaxNode();
      if (raiseStatement.expressions().isEmpty()) {
        return;
      }
      Expression raisedExpression = raiseStatement.expressions().get(0);
      Symbol symbol = null;
      if (raisedExpression instanceof HasSymbol hasSymbol) {
        symbol = hasSymbol.symbol();
      } else if (raisedExpression.is(Tree.Kind.CALL_EXPR)) {
        symbol = ((CallExpression) raisedExpression).calleeSymbol();
      }
      if (hasGlobalOrNonLocalUsage(symbol)) {
        return;
      }
      if (!raisedExpression.type().canBeOrExtend(BASE_EXCEPTION) && !raisedExpression.type().canBeOrExtend("type")) {
        // SONARPY-1666: Here we should only exclude type objects that represent Exception types
        ctx.addIssue(raiseStatement, MESSAGE);
        return;
      }
      if (!mayInheritFromBaseException(symbol)) {
        ctx.addIssue(raiseStatement, MESSAGE);
      }
    });
  }

  private static boolean mayInheritFromBaseException(@Nullable Symbol symbol) {
    if (symbol == null) {
      // S3827 will raise the issue in this case
      return true;
    }
    if (BuiltinSymbols.EXCEPTIONS.contains(symbol.fullyQualifiedName()) || BuiltinSymbols.EXCEPTIONS_PYTHON2.contains(symbol.fullyQualifiedName())) {
      return true;
    }
    if (symbol.is(Symbol.Kind.CLASS)) {
      // to handle implicit constructor call like 'raise MyClass'
      return ((ClassSymbol) symbol).canBeOrExtend(BASE_EXCEPTION);
    }
    // to handle other builtins like 'NotImplemented'
    return !BuiltinSymbols.all().contains(symbol.fullyQualifiedName());
  }

  private static boolean hasGlobalOrNonLocalUsage(@Nullable Symbol symbol) {
    return symbol != null && symbol.usages().stream().anyMatch(s -> s.tree().parent().is(Tree.Kind.GLOBAL_STMT, Tree.Kind.NONLOCAL_STMT));
  }
}
