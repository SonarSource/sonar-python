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

import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8906")
public class BeautifulSoupClassListCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TEMPLATE = "Use \"%s()\" with a chained CSS class selector instead.";
  private static final String MESSAGE_SELECT_ONE = MESSAGE_TEMPLATE.formatted("select_one");
  private static final String MESSAGE_SELECT = MESSAGE_TEMPLATE.formatted("select");


  private static final TypeMatcher IS_BS4_PAGE_ELEMENT_INSTANCE = TypeMatchers.isObjectInstanceOf("bs4.element.PageElement");

  private static TypeMatcher bs4Matcher(String suffix) {
    return TypeMatchers.isType("bs4.element." + suffix);
  }

  // Single-result methods: the correct CSS replacement is select_one().
  // Defined on Tag: find. Defined on PageElement: find_parent, find_next, find_next_sibling, find_previous, find_previous_sibling.
  private static final TypeMatcher IS_BS4_SINGLE_RESULT_CALL = TypeMatchers.any(
    bs4Matcher("Tag.find"),
    bs4Matcher("PageElement.find_parent"),
    bs4Matcher("PageElement.find_next"),
    bs4Matcher("PageElement.find_next_sibling"),
    bs4Matcher("PageElement.find_previous"),
    bs4Matcher("PageElement.find_previous_sibling")
  );

  // List-returning methods: the correct CSS replacement is select().
  // Defined on Tag: find_all. Defined on PageElement: find_parents, find_all_next, find_next_siblings, find_all_previous, find_previous_siblings.
  private static final TypeMatcher IS_BS4_LIST_RESULT_CALL = TypeMatchers.any(
    bs4Matcher("Tag.find_all"),
    bs4Matcher("PageElement.find_parents"),
    bs4Matcher("PageElement.find_all_next"),
    bs4Matcher("PageElement.find_next_siblings"),
    bs4Matcher("PageElement.find_all_previous"),
    bs4Matcher("PageElement.find_previous_siblings")
  );

  // Deprecated camelCase aliases; type inference cannot resolve them (they are simple assignments in the stubs).
  // Instead we check: method name is in this set AND qualifier is a bs4.element.Tag instance.
  // BeautifulSoup extends Tag, so isObjectInstanceOf("bs4.element.Tag") covers both.
  private static final Set<String> DEPRECATED_SINGLE_RESULT_NAMES = Set.of(
    "findChild",
    "findParent",
    "findNext",
    "findNextSibling",
    "findPrevious",
    "findPreviousSibling"
  );

  private static final Set<String> DEPRECATED_LIST_RESULT_NAMES = Set.of(
    "findAll",
    "findChildren",
    "findParents",
    "findAllNext",
    "findNextSiblings",
    "findAllPrevious",
    "findPreviousSiblings"
  );


  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, BeautifulSoupClassListCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression call = (CallExpression) ctx.syntaxNode();
    RegularArgument classArg = TreeUtils.argumentByKeyword("class_", call.arguments());
    if (classArg == null || !classArg.expression().is(Tree.Kind.LIST_LITERAL)) {
      return;
    }
    ListLiteral listLiteral = (ListLiteral) classArg.expression();
    if (listLiteral.elements().expressions().size() < 2) {
      return;
    }
    messageForCall(call, ctx).ifPresent(message -> ctx.addIssue(classArg, message));
  }

  /**
   * Returns the appropriate issue message if the call targets a bs4 search method, or empty otherwise.
   * Single-result methods (find, find_parent, …) → MESSAGE_SELECT_ONE.
   * List-returning methods (find_all, find_parents, …) → MESSAGE_SELECT.
   */
  private static Optional<String> messageForCall(CallExpression call, SubscriptionContext ctx) {
    Expression callee = call.callee();
    // Path 1: modern snake_case methods — resolved via type inference
    if (IS_BS4_SINGLE_RESULT_CALL.isTrueFor(callee, ctx)) {
      return Optional.of(MESSAGE_SELECT_ONE);
    }
    if (IS_BS4_LIST_RESULT_CALL.isTrueFor(callee, ctx)) {
      return Optional.of(MESSAGE_SELECT);
    }
    // Path 2: deprecated camelCase aliases — detected by method name + qualifier type
    if (callee instanceof QualifiedExpression qualifiedExpression) {
      return messageForDeprecatedAlias(qualifiedExpression, ctx);
    }
    return Optional.empty();
  }

  private static Optional<String> messageForDeprecatedAlias(QualifiedExpression qualifiedExpression, SubscriptionContext ctx) {
    String name = qualifiedExpression.name().name();
    if (!IS_BS4_PAGE_ELEMENT_INSTANCE.isTrueFor(qualifiedExpression.qualifier(), ctx)) {
      return Optional.empty();
    }
    if (DEPRECATED_SINGLE_RESULT_NAMES.contains(name)) {
      return Optional.of(MESSAGE_SELECT_ONE);
    }
    if (DEPRECATED_LIST_RESULT_NAMES.contains(name)) {
      return Optional.of(MESSAGE_SELECT);
    }
    return Optional.empty();
  }
}
