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
import org.sonar.python.api.tree.CallExpression;
import org.sonar.python.api.tree.ExpressionList;
import org.sonar.python.api.tree.ForStatement;
import org.sonar.python.api.tree.FunctionDef;
import org.sonar.python.api.tree.Name;
import org.sonar.python.api.tree.StringElement;
import org.sonar.python.api.tree.StringLiteral;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;
import org.sonar.python.semantic.Symbol;
import org.sonar.python.semantic.Usage;

@Rule(key = "S1481")
public class UnusedLocalVariableCheck extends PythonSubscriptionCheck {

  private static final Pattern INTERPOLATION_PATTERN = Pattern.compile("\\{(.*?)\\}");

  private static final String MESSAGE = "Remove the unused local variable \"%s\".";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> {
      FunctionDef functionTree = (FunctionDef) ctx.syntaxNode();
      // https://docs.python.org/3/library/functions.html#locals
      if (isCallingLocalsFunction(functionTree)) {
        return;
      }
      Set<String> interpolationIdentifiers = extractStringInterpolationIdentifiers(functionTree);
      for (Symbol symbol : functionTree.localVariables()) {
        if (interpolationIdentifiers.stream().noneMatch(id -> id.contains(symbol.name())) && !"_".equals(symbol.name()) && hasOnlyBindingUsages(symbol)) {
          symbol.usages().stream()
            .filter(usage -> usage.tree().parent() == null || !usage.tree().parent().is(Kind.PARAMETER))
            .filter(usage -> !isTupleDeclaration(usage.tree()))
            .forEach(usage -> ctx.addIssue(usage.tree(), String.format(MESSAGE, symbol.name())));
        }
      }
    });
  }

  private static boolean hasOnlyBindingUsages(Symbol symbol) {
    List<Usage> usages = symbol.usages();
    return usages.stream().noneMatch(usage -> usage.kind() == Usage.Kind.IMPORT)
      && usages.stream().allMatch(Usage::isBindingUsage);
  }

  private static boolean isTupleDeclaration(Tree tree) {
    return tree.ancestors().stream()
      .anyMatch(t -> t.is(Kind.TUPLE)
        || (t.is(Kind.EXPRESSION_LIST) && ((ExpressionList) t).expressions().size() > 1)
        || (t.is(Kind.FOR_STMT) && ((ForStatement) t).expressions().size() > 1 && ((ForStatement) t).expressions().contains(tree)));
  }

  private static boolean isCallingLocalsFunction(FunctionDef functionTree) {
    return functionTree
      .descendants(Kind.CALL_EXPR)
      .map(CallExpression.class::cast)
      .map(CallExpression::callee)
      .anyMatch(callee -> callee.is(Kind.NAME) && "locals".equals(((Name) callee).name()));
  }

  private static Set<String> extractStringInterpolationIdentifiers(FunctionDef functionTree) {
    return functionTree.descendants(Kind.STRING_LITERAL)
      .map(StringLiteral.class::cast)
      .flatMap(str -> str.stringElements().stream())
      .filter(str -> str.prefix().equalsIgnoreCase("f"))
      .map(StringElement::trimmedQuotesValue)
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
