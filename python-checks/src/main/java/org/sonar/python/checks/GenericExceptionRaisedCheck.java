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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RaiseStatement;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.types.InferredType;

import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.EXCEPTION;

@Rule(key = "S112")
public class GenericExceptionRaisedCheck extends PythonSubscriptionCheck {

  private static final Set<String> GENERIC_EXCEPTION_NAMES = new HashSet<>(Arrays.asList(EXCEPTION, BASE_EXCEPTION));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.RAISE_STMT, ctx -> {
      RaiseStatement raise = (RaiseStatement) ctx.syntaxNode();
      List<Expression> expressions = raise.expressions();
      if (expressions.isEmpty()) {
        return;
      }
      Expression expression = expressions.get(0);
      InferredType type = expression.type();
      if (GENERIC_EXCEPTION_NAMES.stream().anyMatch(type::canOnlyBe) || isGenericExceptionClass(expression)) {
        ctx.addIssue(expression, "Replace this generic exception class with a more specific one.");
      }
    });
  }

  private static boolean isGenericExceptionClass(Expression expression) {
    if (expression.is(Kind.NAME)) {
      Symbol symbol = ((Name) expression).symbol();
      return symbol != null && GENERIC_EXCEPTION_NAMES.contains(symbol.fullyQualifiedName());
    }
    return false;
  }

}
