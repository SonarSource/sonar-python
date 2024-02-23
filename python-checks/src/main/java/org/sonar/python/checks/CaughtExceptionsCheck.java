/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
import java.util.function.Predicate;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_CLAUSE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_GROUP_CLAUSE;
import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;
import static org.sonar.plugins.python.api.types.BuiltinTypes.DICT;
import static org.sonar.plugins.python.api.types.BuiltinTypes.LIST;
import static org.sonar.plugins.python.api.types.BuiltinTypes.SET;
import static org.sonar.plugins.python.api.types.BuiltinTypes.TUPLE;

@Rule(key = "S5708")
public class CaughtExceptionsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Change this expression to be a class deriving from BaseException or a tuple of such classes.";
  private static final Set<String> NON_COMPLIANT_TYPES = new HashSet<>(Arrays.asList(LIST, SET, DICT));
  public static final String QUICK_FIX_MESSAGE_FORMAT = "Make \"%s\" deriving from \"Exception\"";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(EXCEPT_CLAUSE, CaughtExceptionsCheck::checkExceptClause);
    context.registerSyntaxNodeConsumer(EXCEPT_GROUP_CLAUSE, CaughtExceptionsCheck::checkExceptClause);
  }

  private static void checkExceptClause(SubscriptionContext ctx) {
    Expression exception = ((ExceptClause) ctx.syntaxNode()).exception();
    if (exception == null) {
      return;
    }

    TreeUtils.flattenTuples(exception).forEach(expression -> {
      var expressionSymbolOpt = TreeUtils.getSymbolFromTree(expression);
      var notInheritsFromBaseException = expressionSymbolOpt
        .filter(Predicate.not(CaughtExceptionsCheck::inheritsFromBaseException))
        .isPresent();
      if (!canBeOrExtendBaseException(expression.type()) || notInheritsFromBaseException) {
        var issue = ctx.addIssue(expression, MESSAGE);
        expressionSymbolOpt.ifPresent(symbol -> addQuickFix(issue, symbol));
      }
    });
  }

  private static void addQuickFix(PreciseIssue issue, Symbol symbol) {
    symbol.usages()
      .stream()
      .filter(Usage::isBindingUsage)
      .findFirst()
      .map(Usage::tree)
      .map(Tree::parent)
      .map(TreeUtils.toInstanceOfMapper(ClassDef.class))
      .ifPresent(classDef -> {
        Tree insertAfter = classDef.name();
        String insertingText = "(Exception)";

        Token leftPar = classDef.leftPar();
        if (leftPar != null) {
          ArgList args = classDef.args();
          if (args == null) {
            insertAfter = leftPar;
            insertingText = "Exception";
          } else {
            insertAfter = args;
            insertingText = ", Exception";
          }
        }

        issue.addQuickFix(PythonQuickFix.newQuickFix(String.format(QUICK_FIX_MESSAGE_FORMAT, classDef.name().name()))
          .addTextEdit(TextEditUtils.insertAfter(insertAfter, insertingText))
          .build());
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
    if (type.canBeOrExtend("type")) {
      // SONARPY-1666: Here we should only exclude type objects that represent Exception types
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
