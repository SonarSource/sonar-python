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
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;
import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_CLAUSE;
import static org.sonar.plugins.python.api.tree.Tree.Kind.EXCEPT_GROUP_CLAUSE;
import static org.sonar.plugins.python.api.types.BuiltinTypes.BASE_EXCEPTION;

@Rule(key = "S5708")
public class CaughtExceptionsCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Change this expression to be a class deriving from BaseException or a tuple of such classes.";
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
      if (!canBeOrExtendBaseException(expression, ctx) || notInheritsFromBaseException) {
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

  private static boolean canBeOrExtendBaseException(Expression expression, SubscriptionContext ctx) {
    PythonType pythonType = expression.typeV2();
    TriBool isBaseException = ctx.typeChecker().typeCheckBuilder().isInstanceOf("BaseException").check(pythonType);
    TriBool isTuple = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("tuple").check(pythonType);
    TriBool isType = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName("type").check(pythonType);
    return isBaseException != TriBool.FALSE || isTuple != TriBool.FALSE || isType == TriBool.TRUE;
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
