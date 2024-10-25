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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;

@Rule(key = "S6740")
public class PandasReadNoDataTypeCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Provide the \"dtype\" parameter when calling";

  private static final String READ_CSV = "pandas.read_csv";
  private static final String READ_TABLE = "pandas.read_table";

  private TypeCheckBuilder isPandasReadCsv;
  private TypeCheckBuilder isPandasReadTable;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      isPandasReadCsv = ctx.typeChecker().typeCheckBuilder().isTypeWithName(READ_CSV);
      isPandasReadTable = ctx.typeChecker().typeCheckBuilder().isTypeWithName(READ_TABLE);
    });
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, this::checkReadMethodCall);
  }

  private void checkReadMethodCall(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();
    Optional.of(callExpression)
      .filter(this::isReadCall)
      .filter(ce -> TreeUtils.nthArgumentOrKeyword(1, "dtype", ce.arguments()) == null)
      .flatMap(PandasReadNoDataTypeCheck::getNameTree)
      .ifPresent(name -> subscriptionContext.addIssue(name, getMessage(callExpression)));
  }

  private boolean isReadCall(CallExpression callExpression) {
    return Optional.of(callExpression)
      .map(CallExpression::callee)
      .map(Expression::typeV2)
      .filter(this::isPandasReadCall)
      .isPresent();
  }

  private boolean isPandasReadCall(PythonType type) {
    return getPandaReadCallName(type).isPresent();
  }

  private Optional<String> getPandaReadCallName(PythonType type) {
    if(isPandasReadTable.check(type) == TriBool.TRUE) {
      return Optional.of(READ_TABLE);
    } else if (isPandasReadCsv.check(type) == TriBool.TRUE) {
      return Optional.of(READ_CSV);
    } else {
      return Optional.empty();
    }
  }

  private static Optional<Name> getNameTree(CallExpression expression) {
    return Optional.of(expression.callee())
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::name)
      .or(() -> Optional.of(expression.callee())
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class)));
  }

  private  String getMessage(CallExpression ce) {
    return Optional.ofNullable(ce.callee())
      .map(Expression::typeV2)
      .flatMap(this::getPandaReadCallName)
      .map(name -> String.format("%s \"%s\".", MESSAGE, name))
      .orElse("");
  }
}
