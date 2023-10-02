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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S6741")
public class PandasDataFrameToNumpyCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Do not use \"DataFrame.values\".";
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
      .filter("pandas.core.frame.DataFrame.values"::equals)
      .ifPresent(str -> ctx.addIssue(expr.name(), MESSAGE));
  }
}
