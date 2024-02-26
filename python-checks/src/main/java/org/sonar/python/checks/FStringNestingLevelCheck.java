/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import javax.annotation.Nullable;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FormatSpecifier;
import org.sonar.plugins.python.api.tree.FormattedExpression;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6799")
public class FStringNestingLevelCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Do not nest f-strings too deeply.";

  private static final Set<StringElement> visited = new HashSet<>();

  private static final int MAX_DEPTH = 3;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> visited.clear());
    context.registerSyntaxNodeConsumer(Tree.Kind.STRING_ELEMENT, FStringNestingLevelCheck::checkNestingDepthOfFString);
  }

  private static void checkNestingDepthOfFString(SubscriptionContext ctx) {
    if (!supportsTypeParameterSyntax(ctx)) {
      return;
    }
    StringElement element = (StringElement) ctx.syntaxNode();
    if (isFStringNestedTooDeep(element, 0)) {
      ctx.addIssue(element, MESSAGE);
    }
  }

  private static boolean isFStringNestedTooDeep(StringElement element, final int count) {
    if (!visited.contains(element) && element.isInterpolated()) {
      visited.add(element);
      int updatedCount = count + 1;
      if (updatedCount >= MAX_DEPTH) {
        return true;
      }
      return areFormattedExpressionsNestedTooDeep(element.formattedExpressions(), updatedCount);
    }
    return false;
  }

  private static boolean areFormattedExpressionsNestedTooDeep(List<FormattedExpression> formattedExpressions, int updatedCount) {
    for (FormattedExpression formattedExpression : formattedExpressions) {
      if (isTheNestingTooDeepInExpression(formattedExpression.expression(), updatedCount) ||
        isTheNestingTooDeepInFormatSpecifier(formattedExpression.formatSpecifier(), updatedCount)) {
        return true;
      }
    }
    return false;
  }

  private static boolean isTheNestingTooDeepInExpression(Expression expression, int updatedCount) {
    return Optional.of(expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .map(StringLiteral::stringElements)
      .map(Collection::stream)
      .map(elements -> elements
        .anyMatch(sElement -> isFStringNestedTooDeep(sElement, updatedCount)))
      .orElse(false);
  }

  private static boolean isTheNestingTooDeepInFormatSpecifier(@Nullable FormatSpecifier formatSpecifier, int updatedCount) {
    return Optional.ofNullable(formatSpecifier)
      .map(FormatSpecifier::formatExpressions)
      .map(formattedExpressions -> areFormattedExpressionsNestedTooDeep(formattedExpressions, updatedCount))
      .orElse(false);
  }

  private static boolean supportsTypeParameterSyntax(SubscriptionContext ctx) {
    PythonVersionUtils.Version required = PythonVersionUtils.Version.V_312;

    // All versions must be greater than or equal to the required version.
    return ctx.sourcePythonVersions().stream()
      .allMatch(version -> version.compare(required.major(), required.minor()) >= 0);
  }
}
