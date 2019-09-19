/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionListTree;
import org.sonar.python.api.tree.PyForStatementTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyStringElementTree;
import org.sonar.python.api.tree.PyStringLiteralTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.TreeSymbol;
import org.sonar.python.semantic.Usage;

@Rule(key = "S1481")
public class UnusedLocalVariableCheck extends PythonSubscriptionCheck {

  private static final Pattern INTERPOLATION_PATTERN = Pattern.compile("\\{(.*?)\\}");

  private static final String MESSAGE = "Remove the unused local variable \"%s\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> {
      PyFunctionDefTree functionTree = (PyFunctionDefTree) ctx.syntaxNode();
      // https://docs.python.org/3/library/functions.html#locals
      if (isCallingLocalsFunction(functionTree)) {
        return;
      }
      Set<String> interpolationIdentifiers = extractStringInterpolationIdentifiers(functionTree);
      for (TreeSymbol symbol : functionTree.localVariables()) {
        if (interpolationIdentifiers.stream().noneMatch(id -> id.contains(symbol.name())) && !"_".equals(symbol.name()) && hasOnlyBindingUsages(symbol)) {
          symbol.usages().stream()
            .filter(usage -> usage.tree().parent() == null || !usage.tree().parent().is(Kind.PARAMETER))
            .filter(usage -> !isTupleDeclaration(usage.tree()))
            .forEach(usage -> ctx.addIssue(usage.tree(), String.format(MESSAGE, symbol.name())));
        }
      }
    });
  }

  private static boolean hasOnlyBindingUsages(TreeSymbol symbol) {
    List<Usage> usages = symbol.usages();
    return usages.stream().noneMatch(usage -> usage.kind() == Usage.Kind.IMPORT)
      && usages.stream().allMatch(Usage::isBindingUsage);
  }

  private static boolean isTupleDeclaration(Tree tree) {
    return tree.ancestors().stream()
      .anyMatch(t -> t.is(Kind.TUPLE)
        || (t.is(Kind.EXPRESSION_LIST) && ((PyExpressionListTree) t).expressions().size() > 1)
        || t.is(Kind.FOR_STMT) && ((PyForStatementTree) t).expressions().size() > 1 && ((PyForStatementTree) t).expressions().contains(tree));
  }

  private static boolean isCallingLocalsFunction(PyFunctionDefTree functionTree) {
    return functionTree
      .descendants(Kind.CALL_EXPR)
      .map(PyCallExpressionTree.class::cast)
      .map(PyCallExpressionTree::callee)
      .anyMatch(callee -> callee.is(Kind.NAME) && "locals".equals(((PyNameTree) callee).name()));
  }

  private static Set<String> extractStringInterpolationIdentifiers(PyFunctionDefTree functionTree) {
    return functionTree.descendants(Kind.STRING_LITERAL)
      .map(PyStringLiteralTree.class::cast)
      .flatMap(str -> str.stringElements().stream())
      .filter(str -> str.prefix().equalsIgnoreCase("f"))
      .map(PyStringElementTree::trimmedQuotesValue)
      .flatMap(UnusedLocalVariableCheck::extractInterpolations)
      .collect(Collectors.toSet());
  }

  private static Stream<String> extractInterpolations(String str) {
    Matcher matcher = INTERPOLATION_PATTERN.matcher(str);
    List<String> identifiers = new ArrayList<>();
    while (matcher.find()) {
      identifiers.add(matcher.group(1));
    }
    return identifiers.stream();
  }
}
