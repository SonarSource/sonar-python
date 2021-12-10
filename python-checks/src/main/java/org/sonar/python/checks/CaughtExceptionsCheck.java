/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_CLAUSE;
import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.DICT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.LIST;
import static org.sonar.plugins.python.api.types.BuiltinTypes.SET;
import static org.sonar.plugins.python.api.types.BuiltinTypes.TUPLE;

@Rule(key = "S5708")
public class CaughtExceptionsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Change this expression to be a class deriving from BaseException or a tuple of such classes.";
  private static final Set<String> NON_COMPLIANT_TYPES = new HashSet<>(Arrays.asList(LIST, SET, DICT));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(EXCEPT_CLAUSE, ctx -> {
      Expression exception = ((ExceptClause) ctx.syntaxNode()).exception();
      if (exception == null) {
        return;
      }
      TreeUtils.flattenTuples(exception).forEach(expression -> {
        if (!canBeOrExtendBaseException(expression.type()) ||
          ((expression instanceof HasSymbol) && !inheritsFromBaseException(((HasSymbol) expression).symbol()))) {

          ctx.addIssue(expression, MESSAGE);
        }
      });
    });
  }

  private static boolean canBeOrExtendBaseException(InferredType type) {
    if (NON_COMPLIANT_TYPES.stream().anyMatch(type::canOnlyBe)) {
      // due to some limitations in type inference engine,
      // type.canBeOrExtend("list" | "set" | "dict") returns true
      return false;
    }
    if (type.canBeOrExtend(TUPLE)) {
      // avoid FP on variables holding a tuple: SONARPY-713
      return true;
    }
    return type.canBeOrExtend(BASE_EXCEPTION);
  }

  private static boolean inheritsFromBaseException(@Nullable Symbol symbol) {
    if (symbol == null || symbol.kind() != CLASS) {
      // to avoid FP in case of e.g. OSError
      return true;
    }
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    return classSymbol.canBeOrExtend(BASE_EXCEPTION);
  }
}
