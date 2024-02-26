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

import java.util.Map;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6740")
public class PandasReadNoDataTypeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Provide the \"dtype\" parameter when calling";

  private static final String READ_CSV = "pandas.read_csv";
  private static final String READ_TABLE = "pandas.read_table";

  private static final Map<String, String> READ_METHODS = Map.of(
    READ_CSV, READ_CSV,
    READ_TABLE, READ_TABLE,
    "pandas.io.parsers.readers.read_csv", READ_CSV,
    "pandas.io.parsers.readers.read_table", READ_TABLE
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PandasReadNoDataTypeCheck::checkReadMethodCall);
  }

  private static void checkReadMethodCall(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();
    Optional.of(callExpression)
      .filter(PandasReadNoDataTypeCheck::isReadCall)
      .filter(ce -> TreeUtils.nthArgumentOrKeyword(1, "dtype", ce.arguments()) == null)
      .flatMap(PandasReadNoDataTypeCheck::getNameTree)
      .ifPresent(name -> subscriptionContext.addIssue(name, getMessage(callExpression)));
  }

  private static boolean isReadCall(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .filter(PandasReadNoDataTypeCheck::isPandasReadCall)
      .isPresent();
  }

  private static Optional<Name> getNameTree(CallExpression expression) {
    return Optional.of(expression.callee())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::name)
      .or(() -> Optional.of(expression.callee())
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class)));
  }

  private static boolean isPandasReadCall(String fqn) {
    return READ_METHODS.containsKey(fqn);
  }

  private static String getMessage(CallExpression ce) {
    return Optional.ofNullable(ce.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .map(READ_METHODS::get)
      .map(name -> String.format("%s \"%s\".", MESSAGE, name))
      .orElse("");
  }
}
