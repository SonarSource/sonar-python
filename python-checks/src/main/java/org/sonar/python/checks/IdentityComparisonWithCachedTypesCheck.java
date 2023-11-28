/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

import static java.util.Arrays.asList;

// https://jira.sonarsource.com/browse/RSPEC-5795
// https://jira.sonarsource.com/browse/SONARPY-673
@Rule(key = "S5795")
public class IdentityComparisonWithCachedTypesCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE_IS = "Replace this \"is\" operator with \"==\"; identity operator is not reliable here.";
  private static final String MESSAGE_IS_NOT = "Replace this \"is not\" operator with \"!=\"; identity operator is not reliable here.";
  public static final String IS_QUICK_FIX_MESSAGE = "Replace with \"==\"";
  public static final String IS_NOT_QUICK_FIX_MESSAGE = "Replace with \"!=\"";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.IS, IdentityComparisonWithCachedTypesCheck::checkIsComparison);
  }

  private static void checkIsComparison(SubscriptionContext ctx) {
    IsExpression isExpr = (IsExpression) ctx.syntaxNode();

    // Comparison to none is checked by S5727
    if (isComparisonToNone(isExpr)) {
      return;
    }

    if (isUnsuitableOperand(isExpr.leftOperand()) || isUnsuitableOperand(isExpr.rightOperand())) {
      var notToken = isExpr.notToken();
      if (notToken == null) {
        var quickFix = PythonQuickFix.newQuickFix(IS_QUICK_FIX_MESSAGE)
          .addTextEdit(TextEditUtils.replace(isExpr.operator(), "=="))
          .build();

        var issue = ctx.addIssue(isExpr.operator(), MESSAGE_IS);
        issue.addQuickFix(quickFix);
      } else {
        var quickFix = PythonQuickFix.newQuickFix(IS_NOT_QUICK_FIX_MESSAGE)
          .addTextEdit(TextEditUtils.replace(isExpr.operator(), "!="))
          .addTextEdit(TextEditUtils.removeUntil(notToken, isExpr.rightOperand()))
          .build();

        var issue = ctx.addIssue(isExpr.operator(), notToken, MESSAGE_IS_NOT);
        issue.addQuickFix(quickFix);
      }
    }
  }

  private static boolean isComparisonToNone(IsExpression isExpr) {
    return CheckUtils.isNone(isExpr.leftOperand().type()) || CheckUtils.isNone(isExpr.rightOperand().type());
  }

  /**
   * Fully qualified names of constructors and functions that would are guaranteed to create fresh objects with
   * references not shared anywhere else.
   *
   * If a reference arises from an call to one of these functions, and it does not escape anywhere else, then an
   * <code>is</code>-comparison with such a reference will always return <code>False</code>.
   *
   * Note that these are fully qualified names of (value-level) expressions, not types.
   */
  private static final Set<String> FQNS_CONSTRUCTORS_RETURNING_UNIQUE_REF = new HashSet<>(
    asList("frozenset", "bytes", "int", "float", "str", "tuple", "hash"));

  /**
   * Names of types that usually should not be compared with <code>is</code>.
   *
   * Note that these are names of types, not `fqn`s of expressions.
   */
  private static final Set<String> NAMES_OF_TYPES_UNSUITABLE_FOR_COMPARISON = new HashSet<>(
    asList("frozenset", "bytes", "int", "float", "tuple"));

  /**
   * Checks whether an expression is either of a type that should generally be compared with <code>==</code>, or
   * whether it holds a non-escaping reference that stems from a function that guarantees that the
   * returned reference is not shared anywhere.
   */
  private static boolean isUnsuitableOperand(Expression expr) {
    InferredType tpe = expr.type();
    if (isUnsuitableType(tpe)) {
      if (expr.is(Tree.Kind.NAME)) {
        Symbol symb = ((Name) expr).symbol();
        // Impossible to cover. If the type could be inferred, then the `symb` cannot be null.
        return symb != null && !isVariableThatCanEscape(symb);
      } else {
        return true;
      }
    }
    if (expr.is(Tree.Kind.STRING_LITERAL)) {
      return true;
    } else if (expr.is(Tree.Kind.CALL_EXPR)) {
      Symbol calleeSymbol = ((CallExpression) expr).calleeSymbol();
      if (calleeSymbol != null) {
        String calleeFqn = calleeSymbol.fullyQualifiedName();
        return FQNS_CONSTRUCTORS_RETURNING_UNIQUE_REF.contains(calleeFqn);
      }
    }
    return false;
  }

  private static boolean isUnsuitableType(InferredType tpe) {
    return NAMES_OF_TYPES_UNSUITABLE_FOR_COMPARISON.stream().anyMatch(tpe::canOnlyBe);
  }

  /**
   * Checks whether a variable is used anywhere except at the <code>is</code>-comparison site and in the defining
   * assignment. In such cases, we say that the reference to the value of the variable could have escaped,
   * and don't raise an issue. We do not check whether it could actually arrive in the other operand.
   */
  private static boolean isVariableThatCanEscape(Symbol symb) {
    List<Usage> usages = symb.usages();
    if (usages.size() > 2) {
      // Check whether there are any usages except assignments and `is`-comparisons.
      // (Assignments override, they don't let the reference escape. Comparisons with `is` also don't let it escape,
      // because they only compare, they don't copy the reference anywhere.)
      return usages.stream().anyMatch(u -> !u.isBindingUsage() &&
        TreeUtils.firstAncestorOfKind(u.tree(), Tree.Kind.IS) == null);
    }
    // Must be definition and usage within <code>is</code>-comparison itself.
    return false;
  }
}
