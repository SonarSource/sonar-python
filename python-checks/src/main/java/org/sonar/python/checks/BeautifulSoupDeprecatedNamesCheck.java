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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8900")
public class BeautifulSoupDeprecatedNamesCheck extends PythonSubscriptionCheck {

  // All deprecated methods and attributes are defined on PageElement (the base class),
  // so using isObjectInstanceOf covers BeautifulSoup, Tag, and their subclasses.
  private static final TypeMatcher BS4_MATCHER = TypeMatchers.isObjectInstanceOf("bs4.element.PageElement");

  // Maps deprecated method names to their modern equivalents
  private static final Map<String, String> DEPRECATED_METHODS;
  static {
    Map<String, String> m = new LinkedHashMap<>();
    m.put("findAll", "find_all");
    m.put("findChild", "find");
    m.put("findChildren", "find_all");
    m.put("findNext", "find_next");
    m.put("findAllNext", "find_all_next");
    m.put("findPrevious", "find_previous");
    m.put("findAllPrevious", "find_all_previous");
    m.put("findNextSibling", "find_next_sibling");
    m.put("findNextSiblings", "find_next_siblings");
    m.put("findPreviousSibling", "find_previous_sibling");
    m.put("findPreviousSiblings", "find_previous_siblings");
    m.put("findParent", "find_parent");
    m.put("findParents", "find_parents");
    m.put("replaceWith", "replace_with");
    m.put("getText", "get_text");
    DEPRECATED_METHODS = Collections.unmodifiableMap(m);
  }

  // Maps deprecated attribute names to their modern equivalents
  private static final Map<String, String> DEPRECATED_ATTRS;
  static {
    Map<String, String> m = new LinkedHashMap<>();
    m.put("nextSibling", "next_sibling");
    m.put("previousSibling", "previous_sibling");
    DEPRECATED_ATTRS = Collections.unmodifiableMap(m);
  }

  // Modern find-family method names that accept the text= keyword argument.
  // replaceWith/replace_with and getText/get_text are intentionally excluded.
  private static final Set<String> FIND_FAMILY_MODERN_METHODS;
  static {
    Set<String> s = new LinkedHashSet<>();
    s.add("find");
    s.add("find_all");
    s.add("find_next");
    s.add("find_all_next");
    s.add("find_previous");
    s.add("find_all_previous");
    s.add("find_next_sibling");
    s.add("find_next_siblings");
    s.add("find_previous_sibling");
    s.add("find_previous_siblings");
    FIND_FAMILY_MODERN_METHODS = Collections.unmodifiableSet(s);
  }

  // Union of modern find-family names and all deprecated names that map to one of them.
  // Derived to avoid duplicating the deprecated keys from DEPRECATED_METHODS.
  private static final Set<String> FIND_FAMILY_METHODS;
  static {
    LinkedHashSet<String> s = Stream.concat(
      FIND_FAMILY_MODERN_METHODS.stream(),
      DEPRECATED_METHODS.entrySet().stream()
        .filter(e -> FIND_FAMILY_MODERN_METHODS.contains(e.getValue()))
        .map(Map.Entry::getKey)
    ).collect(Collectors.toCollection(LinkedHashSet::new));
    FIND_FAMILY_METHODS = Collections.unmodifiableSet(s);
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, BeautifulSoupDeprecatedNamesCheck::checkCallExpression);
    context.registerSyntaxNodeConsumer(Tree.Kind.QUALIFIED_EXPR, BeautifulSoupDeprecatedNamesCheck::checkQualifiedExpression);
  }

  private static void checkCallExpression(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    if (!(callExpression.callee() instanceof QualifiedExpression qualifiedExpr)) {
      return;
    }

    String methodName = qualifiedExpr.name().name();
    Expression receiver = qualifiedExpr.qualifier();

    if (!BS4_MATCHER.isTrueFor(receiver, ctx)) {
      return;
    }

    // Check for deprecated method name
    // We need to use the name as findAll FQN resolves to find_all
    String modernMethod = DEPRECATED_METHODS.get(methodName);
    if (modernMethod != null) {
      PreciseIssue issue = ctx.addIssue(qualifiedExpr.name(),
        String.format("Replace the deprecated '%s()' method with '%s()'.", methodName, modernMethod));
      issue.addQuickFix(PythonQuickFix.newQuickFix(
        "Replace '%s()' with '%s()'".formatted(methodName, modernMethod),
        TextEditUtils.replace(qualifiedExpr.name(), modernMethod)
      ));
    }

    // Check for deprecated text= keyword argument (in any find-family method)
    if (FIND_FAMILY_METHODS.contains(methodName)) {
      checkDeprecatedTextKeyword(ctx, callExpression.arguments());
    }
  }

  private static void checkDeprecatedTextKeyword(SubscriptionContext ctx, List<Argument> arguments) {
    RegularArgument textArg = TreeUtils.argumentByKeyword("text", arguments);
    if (textArg != null && textArg.keywordArgument() != null) {
      PreciseIssue issue = ctx.addIssue(textArg.keywordArgument(),
        "Replace the deprecated 'text' keyword argument with 'string'.");
      issue.addQuickFix(PythonQuickFix.newQuickFix(
        "Replace 'text' with 'string'",
        TextEditUtils.replace(textArg.keywordArgument(), "string")
      ));
    }
  }

  private static void checkQualifiedExpression(SubscriptionContext ctx) {
    QualifiedExpression qualifiedExpr = (QualifiedExpression) ctx.syntaxNode();

    // Skip if this is the callee of a call expression — that case is handled by checkCallExpression
    if (qualifiedExpr.parent() instanceof CallExpression parentCall
      && parentCall.callee() == qualifiedExpr) {
      return;
    }

    String attrName = qualifiedExpr.name().name();
    String modernAttr = DEPRECATED_ATTRS.get(attrName);
    if (modernAttr == null) {
      return;
    }

    Expression qualifier = qualifiedExpr.qualifier();
    if (BS4_MATCHER.isTrueFor(qualifier, ctx)) {
      PreciseIssue issue = ctx.addIssue(qualifiedExpr.name(),
        String.format("Replace the deprecated '%s' attribute with '%s'.", attrName, modernAttr));
      issue.addQuickFix(PythonQuickFix.newQuickFix(
        "Replace '%s' with '%s'".formatted(attrName, modernAttr),
        TextEditUtils.replace(qualifiedExpr.name(), modernAttr)
      ));
    }
  }
}
