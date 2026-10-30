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
package org.sonar.python.checks.utils;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.tree.TreeUtils;

public final class ReachingValues {
  private ReachingValues() {
  }

  @Nullable
  public static Expression singleValue(Name name, SubscriptionContext ctx) {
    var values = ctx.valuesAtLocation(name);
    if (values.isEmpty()) {
      // Data flow may be unavailable (for example, in scopes containing try). A unique
      // V2 assignment still provides a value, including successive links in an alias chain.
      return Optional.ofNullable(name.symbolV2())
        .flatMap(SymbolV2::getSingleBindingUsage)
        .filter(usage -> usage.kind() == UsageV2.Kind.ASSIGNMENT_LHS)
        .map(UsageV2::tree)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
        .map(ReachingValues::assignedValue)
        .orElse(null);
    }
    return values.size() == 1 ? values.iterator().next() : null;
  }

  /**
   * Expands assignments, aliases, parentheses and conditional branches to possible values.
   * Unresolved expressions and cycles are retained. Shared acyclic paths are expanded
   * only once, so converging aliases do not introduce unresolved values.
   */
  public static Set<Expression> resolvedValues(Expression expression, SubscriptionContext ctx) {
    Set<Expression> result = new HashSet<>();
    Set<Expression> visited = new HashSet<>();
    Set<Expression> activePath = new HashSet<>();
    var pending = new ArrayDeque<ResolutionStep>();
    pending.push(new ResolutionStep(expression, false));
    while (!pending.isEmpty()) {
      ResolutionStep step = pending.pop();
      Expression current = Expressions.removeParentheses(step.expression());
      if (step.finished()) {
        activePath.remove(current);
      } else if (activePath.contains(current)) {
        result.add(current);
      } else if (visited.add(current)) {
        Collection<Expression> values = nextValues(current, ctx);
        if (values.isEmpty()) {
          result.add(current);
        } else {
          activePath.add(current);
          pending.push(new ResolutionStep(current, true));
          values.forEach(value -> pending.push(new ResolutionStep(value, false)));
        }
      }
    }
    return result;
  }

  private static Collection<Expression> nextValues(Expression expression, SubscriptionContext ctx) {
    if (expression instanceof Name name) {
      var values = ctx.valuesAtLocation(name);
      if (!values.isEmpty()) {
        return values;
      }
      Expression value = singleValue(name, ctx);
      return value == null ? List.of() : List.of(value);
    }
    if (expression instanceof ConditionalExpression conditional) {
      return List.of(conditional.trueExpression(), conditional.falseExpression());
    }
    return List.of();
  }

  private record ResolutionStep(Expression expression, boolean finished) {
  }

  @Nullable
  private static Expression assignedValue(Name binding) {
    if (binding.parent() instanceof AnnotatedAssignment assignment) {
      return assignment.assignedValue();
    }
    if (binding.parent() instanceof ExpressionList lhs && lhs.expressions().size() == 1
      && lhs.parent() instanceof AssignmentStatement assignment) {
      return assignment.assignedValue();
    }
    return null;
  }
}
