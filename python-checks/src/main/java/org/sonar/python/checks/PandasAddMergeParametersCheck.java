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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6735")
public class PandasAddMergeParametersCheck extends PythonSubscriptionCheck {

  enum Parameters {
    HOW("how", 2, 1, 2),
    ON("on", 1, 2, 3),
    VALIDATE("validate", 6, 11, 12);

    public String getKeyword() {
      return keyword;
    }

    public int getJoinPosition() {
      return joinPosition;
    }

    public int getDataFrameMergePosition() {
      return dataFrameMergePosition;
    }

    public int getPandasMergePosition() {
      return pandasMergePosition;
    }

    final String keyword;
    final int joinPosition;
    final int dataFrameMergePosition;
    final int pandasMergePosition;

    Parameters(String keyword, int joinPosition, int dataFrameMergePosition, int pandasMergePosition) {
      this.keyword = keyword;
      this.joinPosition = joinPosition;
      this.dataFrameMergePosition = dataFrameMergePosition;
      this.pandasMergePosition = pandasMergePosition;
    }
  }

  private static final Set<String> MERGE_METHODS = Set.of(
    "pandas.core.frame.DataFrame.merge",
    "pandas.core.reshape.merge.merge",
    "pandas.core.frame.DataFrame.join");

  private static final List<String> messages = List.of(
    "The '%s' parameter of the %s should be specified.",
    "The '%s' and '%s' parameters of the %s should be specified.",
    "The '%s', '%s' and '%s' parameters of the %s should be specified.");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PandasAddMergeParametersCheck::verifyMergeCallParameters);
  }

  private static void verifyMergeCallParameters(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(MERGE_METHODS::contains)
      .ifPresent(fqn -> missingArguments(fqn, ctx, callExpression));
  }

  private static void missingArguments(String fullyQualifiedName, SubscriptionContext ctx, CallExpression callExpression) {
    List<String> parameters = new ArrayList<>();
    if (argumentIsMissing(fullyQualifiedName, Parameters.HOW, callExpression.arguments())) {
      parameters.add(Parameters.HOW.getKeyword());
    }
    if (argumentIsMissing(fullyQualifiedName, Parameters.ON, callExpression.arguments())) {
      parameters.add(Parameters.ON.getKeyword());
    }
    if (argumentIsMissing(fullyQualifiedName, Parameters.VALIDATE, callExpression.arguments())) {
      parameters.add(Parameters.VALIDATE.getKeyword());
    }
    if (!parameters.isEmpty()) {
      parameters.add(fullyQualifiedName.substring(fullyQualifiedName.lastIndexOf('.') + 1));
      ctx.addIssue(callExpression, String.format(messages.get(parameters.size() - 2), parameters.toArray()));
    }
  }

  private static boolean argumentIsMissing(String fullyQualfiedName, Parameters keyword, List<Argument> arguments) {
    switch (keyword) {
      case HOW:
        return TreeUtils.nthArgumentOrKeyword(getPosition(fullyQualfiedName, Parameters.HOW), Parameters.HOW.getKeyword(), arguments) == null;
      case ON:
        return TreeUtils.nthArgumentOrKeyword(getPosition(fullyQualfiedName, Parameters.ON), Parameters.ON.getKeyword(), arguments) == null;
      default: // case VALIDATE
        return TreeUtils.nthArgumentOrKeyword(getPosition(fullyQualfiedName, Parameters.VALIDATE), Parameters.VALIDATE.getKeyword(), arguments) == null;
    }
  }

  private static int getPosition(String fullyQualifiedName, Parameters parameter) {
    if ("pandas.core.frame.DataFrame.join".equals(fullyQualifiedName)) {
      return parameter.getJoinPosition();
    } else if ("pandas.core.frame.DataFrame.merge".equals(fullyQualifiedName)) {
      return parameter.getDataFrameMergePosition();
    }
    return parameter.getPandasMergePosition();
  }
}
