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
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S6741")
public class PandasDataFrameToNumpyCheck extends PythonSubscriptionCheck {

  private static final String DATAFRAME_VALUES_FQN = "pandas.core.frame.DataFrame.values";
  private static final String MESSAGE = "Do not use \"DataFrame.values\".";
  private static final String QUICK_FIX_MESSAGE = "Replace with \"DataFrame.to_numpy()\"";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.QUALIFIED_EXPR, this::checkForDataFrameValues);
  }

  private void checkForDataFrameValues(SubscriptionContext ctx) {
    QualifiedExpression expr = (QualifiedExpression) ctx.syntaxNode();
    if ((Optional.of(expr)
      .filter(ex -> "values".equals(ex.name().name())).isEmpty())) {
      return;
    }

    expr.qualifier().type()
      .resolveMember("values")
      .map(Symbol::fullyQualifiedName)
      .filter(DATAFRAME_VALUES_FQN::equals)
      .ifPresent(str -> {
        PreciseIssue issue = ctx.addIssue(expr.name(), MESSAGE);
        issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
          .addTextEdit(TextEditUtils.replace(expr.name(), "to_numpy()"))
          .build());
      });
  }
}
