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
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.fixpoint.ReachingDefinitionsAnalysis;

@Rule(key = "S6741")
public class PandasDataFrameToNumpyCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Do not use \"DataFrame.values\".";
  private ReachingDefinitionsAnalysis reachingDefinitionsAnalysis;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> reachingDefinitionsAnalysis = new ReachingDefinitionsAnalysis(ctx.pythonFile()));
    context.registerSyntaxNodeConsumer(Tree.Kind.QUALIFIED_EXPR, this::checkForDataFrameValues);
  }

  private void checkForDataFrameValues(SubscriptionContext ctx) {
    QualifiedExpression expr = (QualifiedExpression) ctx.syntaxNode();
    if ((Optional.of(expr)
      .filter(ex -> "values".equals(ex.name().name())).isEmpty())) {
      return;
    }

    if (expr.qualifier().is(Tree.Kind.NAME)) {
      this.reachingDefinitionsAnalysis.valuesAtLocation((Name) expr.qualifier())
        .stream()
        .filter(exp -> exp.is(Tree.Kind.CALL_EXPR))
        .map(CallExpression.class::cast)
        .filter(ce -> ce.callee().is(Tree.Kind.NAME, Tree.Kind.QUALIFIED_EXPR))
        .map(PandasDataFrameToNumpyCheck::getFullyQualifiedName)
        .filter(Optional::isPresent)
        .map(Optional::get)
        .filter("pandas.DataFrame"::equals).findAny()
        .ifPresent(str -> ctx.addIssue(expr.name(), MESSAGE));
    } else if (expr.qualifier().is(Tree.Kind.CALL_EXPR)) {
      Optional.of((CallExpression) expr.qualifier())
        .map(CallExpression::calleeSymbol)
        .map(Symbol::fullyQualifiedName)
        .filter("pandas.DataFrame"::equals)
        .ifPresent(str -> ctx.addIssue(expr.name(), MESSAGE));
    }
  }

  private static Optional<String> getFullyQualifiedName(CallExpression callExpression) {
    if (callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
      return Optional.of((QualifiedExpression) callExpression.callee())
        .map(QualifiedExpression::name)
        .map(Name::symbol)
        .map(Symbol::fullyQualifiedName);
    } else {
      return Optional.of((Name) callExpression.callee())
        .map(Name::symbol)
        .map(Symbol::fullyQualifiedName);
    }
  }
}
