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

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6903")
public class TimezoneNaiveDatetimeConstructorsCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Don't use `%s` to create this datetime object.";
  private static final Set<String> NON_COMPLIANT_FQNS = Set.of("datetime.datetime.utcnow", "datetime.datetime.utcfromtimestamp");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, TimezoneNaiveDatetimeConstructorsCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext context) {
    CallExpression callExpression = (CallExpression) context.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();

    if (calleeSymbol == null) {
      return;
    }
    String fullyQualifiedName = calleeSymbol.fullyQualifiedName();
    if (fullyQualifiedName == null || !NON_COMPLIANT_FQNS.contains(fullyQualifiedName)) {
      return;
    }
    context.addIssue(callExpression, String.format(MESSAGE, fullyQualifiedName));
  }
}
