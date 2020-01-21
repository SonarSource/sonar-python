/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.BuiltinSymbols;

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
      if (raiseStatement.expressions().get(0).is(Tree.Kind.CALL_EXPR)) {
        CallExpression callExpression = ((CallExpression) raiseStatement.expressions().get(0));
        Symbol calleeSymbol = callExpression.calleeSymbol();
        if (!inheritsFromBaseException(calleeSymbol)) {
          ctx.addIssue(raiseStatement, MESSAGE);
        }
      }
      if (raiseStatement.expressions().get(0).is(Tree.Kind.NAME)) {
        Symbol symbol = ((Name) raiseStatement.expressions().get(0)).symbol();
        if (!inheritsFromBaseException(symbol)) {
          ctx.addIssue(raiseStatement, MESSAGE);
        }
      }
      if (raiseStatement.expressions().get(0).is(Tree.Kind.STRING_LITERAL)) {
        ctx.addIssue(raiseStatement, MESSAGE);
      }
    });
  }

  private boolean inheritsFromBaseException(@Nullable Symbol symbol) {
    if (symbol == null) {
      // S3827 will raise the issue in this case
      return true;
    }
    if (BuiltinSymbols.EXCEPTIONS.contains(symbol.fullyQualifiedName()) || BuiltinSymbols.EXCEPTIONS_PYTHON2.contains(symbol.fullyQualifiedName())) {
      return true;
    }
    if (Symbol.Kind.CLASS.equals(symbol.kind())) {
      // we know it's a class defined in the project
      ClassSymbol classSymbol = (ClassSymbol) symbol;
      if (classSymbol.hasUnresolvedTypeHierarchy()) {
        return true;
      }
      for (Symbol parent : classSymbol.superClasses()) {
        if (inheritsFromBaseException(parent)) {
          return true;
        }
      }
      return false;
    }
    // returns true in case of unknown symbol
    return !BuiltinSymbols.all().contains(symbol.fullyQualifiedName());
  }
}
