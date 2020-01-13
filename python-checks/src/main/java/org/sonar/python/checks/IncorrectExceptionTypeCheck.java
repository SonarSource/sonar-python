/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import java.util.List;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.RegularArgument;
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
    });
  }

  private boolean inheritsFromBaseException(@Nullable Symbol symbol) {
    if (symbol == null) {
      // S3827 will raise the issue in this case
      return true;
    }
    if (BuiltinSymbols.EXCEPTIONS.contains(symbol.name()) || BuiltinSymbols.EXCEPTIONS_PYTHON2.contains(symbol.name())) {
      return true;
    }
    List<Usage> bindingUsages = symbol.usages().stream().filter(Usage::isBindingUsage).collect(Collectors.toList());
    if (bindingUsages.size() > 1) {
      return true;
    }
    if (bindingUsages.size() == 1) {
      Usage usage = bindingUsages.get(0);
      if (usage.kind().equals(Usage.Kind.IMPORT)) {
        return true;
      }
      if (usage.kind().equals(Usage.Kind.CLASS_DECLARATION)) {
        return classInheritsFromBaseException((ClassDef) usage.tree().parent());
      }
    }
    // returns true in case of unknown symbol to avoid FP
    return !BuiltinSymbols.all().contains(symbol.name());
  }

  private boolean classInheritsFromBaseException(ClassDef classDef) {
    ArgList args = classDef.args();
    if (args == null) {
      return false;
    }
    return args.arguments().stream().anyMatch(this::argumentInheritsFromBaseException);
  }

  private boolean argumentInheritsFromBaseException(Argument argument) {
    if (!argument.is(Tree.Kind.REGULAR_ARGUMENT)) {
      // Need type inference to assess further
      return true;
    }
    Expression expression = ((RegularArgument) argument).expression();
    if (expression instanceof HasSymbol) {
      return inheritsFromBaseException(((HasSymbol) expression).symbol());
    }
    return true;
  }
}
