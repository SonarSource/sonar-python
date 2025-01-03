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

import java.util.Map;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S6730")
public class DeprecatedNumpyTypesCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Replace this deprecated \"numpy\" type alias with the builtin type \"%s\".";
  private static final String QUICK_FIX_MESSAGE = "Replace with %s.";
  private static final Map<String, String> TYPE_TO_CHECK = Map.of(
      "numpy.bool", "bool",
      "numpy.int", "int",
      "numpy.float", "float",
      "numpy.complex", "complex",
      "numpy.object", "object",
      "numpy.str", "str",
      "numpy.long", "int",
      "numpy.unicode", "str");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.QUALIFIED_EXPR,
        DeprecatedNumpyTypesCheck::checkForDeprecatedTypesNames);
  }

  private static void checkForDeprecatedTypesNames(SubscriptionContext ctx) {
    QualifiedExpression expression = (QualifiedExpression) ctx.syntaxNode();
    Optional.ofNullable(expression.symbol())
        .map(Symbol::fullyQualifiedName)
        .map(TYPE_TO_CHECK::get)
        .ifPresent(type -> raiseIssue(expression, type, ctx));
  }

  private static void raiseIssue(Tree expression, String replacementType, SubscriptionContext ctx) {
    PreciseIssue issue = ctx.addIssue(expression, String.format(MESSAGE, replacementType));
    PythonQuickFix quickFix = PythonQuickFix.newQuickFix(
        String.format(QUICK_FIX_MESSAGE, replacementType),
        TextEditUtils.replace(expression, replacementType));
    issue.addQuickFix(quickFix);
  }
}
