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

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6735")
public class PandasAddMergeParametersCheck extends PythonSubscriptionCheck {

  enum Keywords {
    HOW("how", 2, 1, 2, "\"inner\"", "\"left\""),
    ON("on", 1, 2, 3, "None", "None"),
    VALIDATE("validate", 6, 11, 12, "\"many_to_many\"", "\"many_to_many\""),
    LEFT_ON("left_on", -1, 3, 4, "None", "None"),
    RIGHT_ON("right_on", -1, 4, 5, "None", "None");

    public String getKeyword() {
      return keyword;
    }

    public int getPositionInJoin() {
      return positionInJoin;
    }

    public int getPositionInDataFrameMerge() {
      return positionInDataFrameMerge;
    }

    public int getPositionInPandasMerge() {
      return positionInPandasMerge;
    }

    final String keyword;
    final int positionInJoin;
    final int positionInDataFrameMerge;
    final int positionInPandasMerge;

    final String defaultValueMerge;
    final String defaultValueJoin;

    String getReplacementText(String fullyQualifiedName) {
      if (DATAFRAME_JOIN_FQN.equals(fullyQualifiedName)) {
        return String.format("%s=%s", this.keyword, this.defaultValueJoin);
      } else {
        return String.format("%s=%s", this.keyword, this.defaultValueMerge);
      }
    }

    int getArgumentPosition(String fullyQualifiedName) {
      if (DATAFRAME_JOIN_FQN.equals(fullyQualifiedName)) {
        return this.getPositionInJoin();
      } else if (DATAFRAME_MERGE_FQN.equals(fullyQualifiedName)) {
        return this.getPositionInDataFrameMerge();
      }
      return this.getPositionInPandasMerge();
    }

    Keywords(String keyword, int positionInJoin, int positionInDataFrameMerge, int positionInPandasMerge, String defaultValueMerge, String defaultValueJoin) {
      this.keyword = keyword;
      this.positionInJoin = positionInJoin;
      this.positionInDataFrameMerge = positionInDataFrameMerge;
      this.positionInPandasMerge = positionInPandasMerge;
      this.defaultValueMerge = defaultValueMerge;
      this.defaultValueJoin = defaultValueJoin;
    }
  }

  private static final String DATAFRAME_JOIN_FQN = "pandas.core.frame.DataFrame.join";
  private static final String DATAFRAME_MERGE_FQN = "pandas.core.frame.DataFrame.merge";
  private static final String PANDAS_MERGE_FQN = "pandas.core.reshape.merge.merge";

  private static final Set<Keywords> ON_KEYWORDS = EnumSet.of(Keywords.ON, Keywords.LEFT_ON, Keywords.RIGHT_ON);
  private static final Set<String> METHODS = Set.of(
    DATAFRAME_JOIN_FQN,
    DATAFRAME_MERGE_FQN,
    PANDAS_MERGE_FQN);

  private static final Map<Integer, String> MESSAGES = Map.of(
    1, "Specify the \"%s\" parameter of this %s.",
    2, "Specify the \"%s\" and \"%s\" parameters of this %s.",
    3, "Specify the \"%s\", \"%s\" and \"%s\" parameters of this %s.");

  private static final String QUICKFIX_MESSAGE = "Add the missing parameters";

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
    List<Keywords> missingKeywords = new ArrayList<>();
    if (isArgumentMissing(fullyQualifiedName, Keywords.HOW, callExpression.arguments())) {
      missingKeywords.add(Keywords.HOW);
    }
    if (ON_KEYWORDS.stream().allMatch(keyword -> isArgumentMissing(fullyQualifiedName, keyword, callExpression.arguments()))) {
      missingKeywords.add(Keywords.ON);
    }
    if (isArgumentMissing(fullyQualifiedName, Keywords.VALIDATE, callExpression.arguments())) {
      missingKeywords.add(Keywords.VALIDATE);
    }
    if (!missingKeywords.isEmpty()) {
      PreciseIssue issue = ctx.addIssue(callExpression, generateMessage(MESSAGES.get(numberOfMissingArguments(missingKeywords)), missingKeywords, fullyQualifiedName));
      issue.addQuickFix(PythonQuickFix
        .newQuickFix(QUICKFIX_MESSAGE)
        .addTextEdit(
          TextEditUtils.insertBefore(callExpression.rightPar(), getReplacementText(fullyQualifiedName, missingKeywords)))
        .build());
    }
  }

  private static boolean isArgumentMissing(String fullyQualfiedName, Keywords keyword, List<Argument> arguments) {
    return Optional.of(keyword)
      .map(kw -> TreeUtils.nthArgumentOrKeyword(kw.getArgumentPosition(fullyQualfiedName), kw.getKeyword(), arguments))
      .isEmpty();
  }

  private static String generateMessage(String message, List<Keywords> missingKeywords, String functionName) {
    List<String> missingKeywordsList = missingKeywords.stream().map(Keywords::getKeyword).collect(Collectors.toList());
    missingKeywordsList.add(functionName.substring(functionName.lastIndexOf('.') + 1));
    return String.format(message, missingKeywordsList.toArray());
  }

  private static int numberOfMissingArguments(List<Keywords> missingKeywords) {
    return missingKeywords.size();
  }

  private static String getReplacementText(String fullyQualifiedName, List<Keywords> missingKeywords) {
    return String.format(", %s", missingKeywords.stream().map(keyword -> keyword.getReplacementText(fullyQualifiedName))
      .collect(Collectors.joining(", ")));
  }
}
