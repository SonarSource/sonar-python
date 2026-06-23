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

import java.util.Map;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8903")
public class BeautifulSoupRawHtmlInsertionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use \"new_tag()\" instead of inserting raw HTML strings.";

  private record MethodSpec(int argIndex, String keyword) {}

  private static final Map<TypeMatcher, MethodSpec> INSERTION_METHODS = Map.of(
    TypeMatchers.isType("bs4.element.PageElement.insert"), new MethodSpec(1, "new_child"),
    TypeMatchers.isType("bs4.element.PageElement.append"), new MethodSpec(0, "tag"),
    TypeMatchers.isType("bs4.element.PageElement.extend"), new MethodSpec(0, "tags")
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, BeautifulSoupRawHtmlInsertionCheck::checkCallExpression);
  }

  private static boolean looksLikeHtmlMarkup(Expression expr) {
    StringLiteral literal = Expressions.extractStringLiteral(expr);
    return literal != null && 
      literal.trimmedQuotesValue().startsWith("<") && 
      literal.trimmedQuotesValue().endsWith(">");
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpr = (CallExpression) ctx.syntaxNode();

    if (!(callExpr.callee() instanceof QualifiedExpression qualifiedExpr)) {
      return;
    }

    INSERTION_METHODS.entrySet().stream()
      .filter(e -> e.getKey().isTrueFor(qualifiedExpr, ctx))
      .map(Map.Entry::getValue)
      .findFirst()
      .flatMap(spec -> TreeUtils.nthArgumentOrKeywordOptional(spec.argIndex(), spec.keyword(), callExpr.arguments()))
      .map(RegularArgument::expression)
      .filter(BeautifulSoupRawHtmlInsertionCheck::looksLikeHtmlMarkup)
      .ifPresent(contentExpr -> ctx.addIssue(qualifiedExpr.name(), MESSAGE));
  }
}
