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

import java.util.Optional;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6725")
public class NumpyIsNanCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Don't perform an equality check against \"numpy.nan\".";
  private static final String QUICK_FIX_MESSAGE_EQUALITY = "Replace this equality check with numpy.isnan().";
  private static final String QUICK_FIX_MESSAGE_INEQUALITY = "Replace this inequality check with !numpy.isnan().";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.COMPARISON, NumpyIsNanCheck::checkForIsNan);
  }

  private static void checkForIsNan(SubscriptionContext ctx) {
    BinaryExpression be = (BinaryExpression) ctx.syntaxNode();
    String value = be.operator().value();
    if (!("==".equals(value) || "!=".equals(value))) {
      return;
    }
    checkOperand(ctx, be.leftOperand(), be);
    checkOperand(ctx, be.rightOperand(), be);
  }

  private static void checkOperand(SubscriptionContext ctx, Expression operand, BinaryExpression be) {
    Optional.ofNullable(getSymbolFromExpression(operand))
      .map(Symbol::fullyQualifiedName)
      .filter("numpy.nan"::equals)
      .ifPresent(fqn -> {
        PreciseIssue issue = ctx.addIssue(be, MESSAGE);
        addQuickFix(issue, operand, be);
      });
  }

  @CheckForNull
  private static Symbol getSymbolFromExpression(Expression symbol) {
    if (symbol.is(Tree.Kind.QUALIFIED_EXPR)) {
      return ((QualifiedExpression) symbol).symbol();
    } else if (symbol.is(Tree.Kind.NAME)) {
      return ((Name) symbol).symbol();
    } else {
      return null;
    }
  }

  private static void addQuickFix(PreciseIssue issue, Expression operand, BinaryExpression be) {
    if (be.leftOperand().equals(operand)) {
      addQuickFix(issue, operand, be.rightOperand(), be);
    } else {
      addQuickFix(issue, operand, be.leftOperand(), be);
    }
  }

  private static void addQuickFix(PreciseIssue issue, Expression nanOperand, Expression otherOperand, BinaryExpression be) {
    Optional.of(nanOperand)
      .filter(op -> op.is(Tree.Kind.QUALIFIED_EXPR))
      .map(QualifiedExpression.class::cast)
      .map(QualifiedExpression::qualifier)
      .filter(quali -> quali.is(Tree.Kind.NAME))
      .map(Name.class::cast)
      .map(Name::name)
      .map(symbName -> addPrefix(be) + symbName + ".isnan(" + TreeUtils.treeToString(otherOperand, true) + ")")
      .ifPresent(replacement -> issue
        .addQuickFix(PythonQuickFix
          .newQuickFix(operatorToMessage(be))
          .addTextEdit(TextEditUtils.replace(be, replacement))
          .build()));
  }

  private static String addPrefix(BinaryExpression be) {
    if ("==".equals(be.operator().value())) {
      return "";
    } else {
      return "!";
    }
  }

  private static String operatorToMessage(BinaryExpression be) {
    if ("==".equals(be.operator().value())) {
      return QUICK_FIX_MESSAGE_EQUALITY;
    } else {
      return QUICK_FIX_MESSAGE_INEQUALITY;
    }
  }
}
