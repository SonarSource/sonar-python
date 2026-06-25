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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8905")
public class BeautifulSoupParserCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Specify the parser to use for \"BeautifulSoup\".";
  private static final TypeMatcher BEAUTIFUL_SOUP_MATCHER = TypeMatchers.isType("bs4.BeautifulSoup");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, BeautifulSoupParserCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    if (!BEAUTIFUL_SOUP_MATCHER.isTrueFor(callExpression.callee(), ctx)) {
      return;
    }

    List<Argument> arguments = callExpression.arguments();

    var features = TreeUtils.nthArgumentOrKeyword(1, "features", arguments);
    var builder = TreeUtils.nthArgumentOrKeyword(2, "builder", arguments);

    if (features == null && builder == null) {
      ctx.addIssue(callExpression.callee(), MESSAGE);
    }
  }
}
