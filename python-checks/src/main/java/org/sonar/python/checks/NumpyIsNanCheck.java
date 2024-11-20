/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Optional;
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

  private static final String MESSAGE = "Don't perform an equality/inequality check against \"numpy.nan\".";
  private static final String QUICK_FIX_MESSAGE_EQUALITY = "Replace this equality check with \"numpy.isnan()\".";
  private static final String QUICK_FIX_MESSAGE_INEQUALITY = "Replace this inequality check with \"not numpy.isnan()\".";

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
    checkOperand(ctx, be.leftOperand(), be.rightOperand(), be);
    checkOperand(ctx, be.rightOperand(), be.leftOperand(), be);
  }

  private static void checkOperand(SubscriptionContext ctx, Expression operand, Expression otherOperand, BinaryExpression be) {
    TreeUtils.getSymbolFromTree(operand)
      .map(Symbol::fullyQualifiedName)
      .filter("numpy.nan"::equals)
      .ifPresent(fqn -> {
        PreciseIssue issue = ctx.addIssue(be, MESSAGE);
        addQuickFix(issue, operand, otherOperand, be);
      });
  }

  private static void addQuickFix(PreciseIssue issue, Expression nanOperand, Expression otherOperand, BinaryExpression be) {
    Optional.of(nanOperand)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::qualifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
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
      return "not ";
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
