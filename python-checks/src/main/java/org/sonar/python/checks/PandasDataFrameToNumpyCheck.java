/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.ReachingValues;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S6741")
public class PandasDataFrameToNumpyCheck extends PythonSubscriptionCheck {

  private static final String DATAFRAME_FQN = "pandas.core.frame.DataFrame";
  private static final TypeMatcher DATAFRAME_MATCHER = TypeMatchers.isObjectSatisfying(TypeMatchers.withFQN(DATAFRAME_FQN));
  private static final TypeMatcher PANDAS_READ_CSV_MATCHER = TypeMatchers.isType("pandas.read_csv");
  private static final TypeMatcher TYPE_HINT_SOURCE_MATCHER = TypeMatchers.hasTypeSource(TypeSource.TYPE_HINT);
  private static final String MESSAGE = "Do not use \"DataFrame.values\".";
  private static final String QUICK_FIX_MESSAGE = "Replace with \"DataFrame.to_numpy()\"";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.QUALIFIED_EXPR, this::checkForDataFrameValues);
  }

  private void checkForDataFrameValues(SubscriptionContext ctx) {
    QualifiedExpression expr = (QualifiedExpression) ctx.syntaxNode();
    if (!"values".equals(expr.name().name())
      || TYPE_HINT_SOURCE_MATCHER.isTrueFor(expr.qualifier(), ctx)
      || !isDataFrameValues(expr, ctx)) {
      return;
    }

    PreciseIssue issue = ctx.addIssue(expr.name(), MESSAGE);
    issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE)
      .addTextEdit(TextEditUtils.replace(expr.name(), "to_numpy()"))
      .build());
  }

  private static boolean isDataFrameValues(QualifiedExpression expr, SubscriptionContext ctx) {
    if (expr.qualifier() instanceof Name name) {
      Expression reachingValue = reachingValue(name, ctx);
      return reachingValue instanceof CallExpression call
        && (DATAFRAME_MATCHER.isTrueFor(call, ctx) || PANDAS_READ_CSV_MATCHER.isTrueFor(call.callee(), ctx));
    }
    return DATAFRAME_MATCHER.isTrueFor(expr.qualifier(), ctx);
  }

  private static Expression reachingValue(Name name, SubscriptionContext ctx) {
    Expression value = ReachingValues.singleValue(name, ctx);
    while (value instanceof Name reachingName) {
      value = ReachingValues.singleValue(reachingName, ctx);
    }
    return value;
  }
}
