/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6734")
public class PandasModifyInPlaceCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Do not use \"inplace=True\" when modifying a dataframe.";

  private static final Set<String> FULLY_QUALIFIED_EXPRESSIONS = Set.of(
    "pandas.core.frame.DataFrame.drop",
    "pandas.core.frame.DataFrame.dropna",
    "pandas.core.frame.DataFrame.drop_duplicates",
    "pandas.core.frame.DataFrame.sort_values",
    "pandas.core.frame.DataFrame.sort_index",
    "pandas.core.frame.DataFrame.eval",
    "pandas.core.frame.DataFrame.query");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PandasModifyInPlaceCheck::checkInplaceParameter);
  }

  private static void checkInplaceParameter(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(FULLY_QUALIFIED_EXPRESSIONS::contains)
      .map(fqn -> TreeUtils.argumentByKeyword("inplace", callExpression.arguments()))
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::name)
      .filter("True"::equals)
      .flatMap(fqn -> Optional.ofNullable(TreeUtils.argumentByKeyword("inplace", callExpression.arguments())))
      .ifPresent(regularArgument -> ctx.addIssue(regularArgument, MESSAGE));
  }
}
