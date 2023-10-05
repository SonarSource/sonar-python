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
import java.util.Map;
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

  enum Keywords {
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

    Keywords(String keyword, int joinPosition, int dataFrameMergePosition, int pandasMergePosition) {
      this.keyword = keyword;
      this.joinPosition = joinPosition;
      this.dataFrameMergePosition = dataFrameMergePosition;
      this.pandasMergePosition = pandasMergePosition;
    }
  }

  private static final Set<String> METHODS = Set.of(
    "pandas.core.frame.DataFrame.merge",
    "pandas.core.reshape.merge.merge",
    "pandas.core.frame.DataFrame.join");

  private static final Map<Integer, String> MESSAGES = Map.of(
    1, "Specify the \"%s\" parameter of this %s.",
    2, "Specify the \"%s\" and \"%s\" parameters of this %s.",
    3, "Specify the \"%s\", \"%s\" and \"%s\" parameters of this %s.");
  // private static final List<String> MESSAGES = List.of(
  // "Specify the \"%s\" parameter of this %s.",
  // "Specify the \"%s\" and \"%s\" parameters of this %s.",
  // "Specify the \"%s\", \"%s\" and \"%s\" parameters of this %s.");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, PandasAddMergeParametersCheck::verifyMergeCallParameters);
  }

  private static void verifyMergeCallParameters(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(METHODS::contains)
      .ifPresent(fqn -> missingArguments(fqn, ctx, callExpression));
  }

  private static void missingArguments(String fullyQualifiedName, SubscriptionContext ctx, CallExpression callExpression) {
    List<String> parameters = new ArrayList<>();
    if (argumentIsMissing(fullyQualifiedName, Keywords.HOW, callExpression.arguments())) {
      parameters.add(Keywords.HOW.getKeyword());
    }
    if (argumentIsMissing(fullyQualifiedName, Keywords.ON, callExpression.arguments())) {
      parameters.add(Keywords.ON.getKeyword());
    }
    if (argumentIsMissing(fullyQualifiedName, Keywords.VALIDATE, callExpression.arguments())) {
      parameters.add(Keywords.VALIDATE.getKeyword());
    }
    if (!parameters.isEmpty()) {
      ctx.addIssue(callExpression, generateMessage(MESSAGES.get(numberOfMissingArguments(parameters)), parameters, fullyQualifiedName));
    }
  }

  private static boolean argumentIsMissing(String fullyQualfiedName, Keywords keyword, List<Argument> arguments) {
    return Optional.of(keyword)
      .map(kw -> TreeUtils.nthArgumentOrKeyword(getArgumentPosition(fullyQualfiedName, kw), kw.getKeyword(), arguments))
      .isEmpty();
  }

  private static int getArgumentPosition(String fullyQualifiedName, Keywords parameter) {
    if ("pandas.core.frame.DataFrame.join".equals(fullyQualifiedName)) {
      return parameter.getJoinPosition();
    } else if ("pandas.core.frame.DataFrame.merge".equals(fullyQualifiedName)) {
      return parameter.getDataFrameMergePosition();
    }
    return parameter.getPandasMergePosition();
  }

  private static String generateMessage(String message, List<String> missingKeywords, String functionName) {
    missingKeywords.add(functionName.substring(functionName.lastIndexOf('.') + 1));
    return String.format(message, missingKeywords.toArray());
  }

  private static int numberOfMissingArguments(List<String> keywords) {
    return keywords.size();
  }
}
