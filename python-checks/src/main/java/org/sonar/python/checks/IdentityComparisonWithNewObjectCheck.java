/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
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
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

import static java.util.Arrays.asList;
import static org.sonar.plugins.python.api.types.BuiltinTypes.NONE_TYPE;

// https://jira.sonarsource.com/browse/RSPEC-5796
// https://jira.sonarsource.com/browse/SONARPY-674
@Rule(key = "S5796")
public class IdentityComparisonWithNewObjectCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE_IS = "Replace this \"is\" operator with \"==\".";
  public static final String IS_QUICK_FIX_MESSAGE = "Replace with \"==\"";
  public static final String IS_NOT_QUICK_FIX_MESSAGE = "Replace with \"!=\"";
  private static final String MESSAGE_IS_NOT = "Replace this \"is not\" operator with \"!=\".";
  private static final String MESSAGE_SECONDARY = "This expression creates a new object every time.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.IS, IdentityComparisonWithNewObjectCheck::checkIsComparison);
  }

  private static void checkIsComparison(SubscriptionContext subscriptionContext) {
    final IsExpression isExpr = (IsExpression) subscriptionContext.syntaxNode();

    // Exit early if we can infer that the types don't match to avoid overlap with RSPEC-3403, RSPEC-5727
    InferredType t1 = isExpr.leftOperand().type();
    InferredType t2 = isExpr.rightOperand().type();
    if (!t1.isIdentityComparableWith(t2) || t1.canOnlyBe(NONE_TYPE) || t2.canOnlyBe(NONE_TYPE)) {
      return;
    }

    // The `if` merely ensures that an issue is reported at most once per operator.
    if (!checkOperand(isExpr.leftOperand(), isExpr, subscriptionContext)) {
      checkOperand(isExpr.rightOperand(), isExpr, subscriptionContext);
    }
  }

  /**
   * Checks a single operand of an <code>is/is not</code>-comparison.
   * Returns <code>true</code> if it finds an issue.
   */
  private static boolean checkOperand(Expression operand, IsExpression isExpr, SubscriptionContext ctx) {
    Optional<List<Tree>> secondariesOpt = findIssueForOperand(operand);
    secondariesOpt.ifPresent(secondaryLocations -> {
      PreciseIssue issue;
      var notToken = isExpr.notToken();
      if (notToken != null) {
        var quickFix = PythonQuickFix.newQuickFix(IS_NOT_QUICK_FIX_MESSAGE)
          .addTextEdit(TextEditUtils.replace(isExpr.operator(), "!="))
          .addTextEdit(TextEditUtils.removeUntil(notToken, isExpr.rightOperand()))
          .build();

        issue = ctx.addIssue(isExpr.operator(), notToken, MESSAGE_IS_NOT);
        issue.addQuickFix(quickFix);
      } else {
        var quickFix = PythonQuickFix.newQuickFix(IS_QUICK_FIX_MESSAGE)
          .addTextEdit(TextEditUtils.replace(isExpr.operator(), "=="))
          .build();
        issue = ctx.addIssue(isExpr.operator(), MESSAGE_IS);
        issue.addQuickFix(quickFix);
      }


      for (Tree secondary : secondaryLocations) {
        issue.secondary(secondary, MESSAGE_SECONDARY);
      }
    });
    return secondariesOpt.isPresent();
  }

  /**
   * Checks whether an operand of an <code>is</code>-comparison is suitable.
   *
   * @param expr the operand
   * @return An empty option if there is no issue, or an option containing the list of relevant secondary expressions.
   */
  private static Optional<List<Tree>> findIssueForOperand(Expression expr) {
    if (instantiatesFreshObject(expr)) {
      // Issue exists, but the list of secondaries is empty.
      return Optional.of(Collections.emptyList());
    } else if (expr.is(Tree.Kind.NAME)) {
      Name name = (Name) expr;
      Symbol symb = name.symbol();
      if (symb != null) {
        Expression rhs = Expressions.singleAssignedValue((Name) expr);
        if (rhs != null && instantiatesFreshObject(rhs) && cannotEscape(symb)) {
          // Issue exists, right hand side of the assignment that defines the variable is the secondary position
          return Optional.of(Collections.singletonList(rhs));
        }
      }
    }
    return Optional.empty();
  }

  /**
   * Fully qualified names of constructors and functions that are guaranteed to return references that aren't shared
   * anywhere else, and are thus unsuitable for a comparison with an <code>is</code>, because the comparison would
   * always return <code>False</code>.
   *
   * Note that these are fully qualified names of (value-level) expressions, not types.
   */
  private static final Set<String> FUNCTIONS_RETURNING_UNIQUE_REF =
    new HashSet<>(asList("dict", "list", "set", "complex"));

  /**
   * Checks whether an expression is guaranteed to instantiate a fresh object with a reference that has not been
   * shared anywhere.
   */
  private static boolean instantiatesFreshObject(Expression expr) {
    switch (expr.getKind()) {
      case DICTIONARY_LITERAL, DICT_COMPREHENSION, LIST_LITERAL, LIST_COMPREHENSION, SET_LITERAL, SET_COMPREHENSION -> {
        return true;
      }
      case CALL_EXPR -> {
        Symbol calleeSymbol = ((CallExpression) expr).calleeSymbol();
        if (calleeSymbol != null) {
          return FUNCTIONS_RETURNING_UNIQUE_REF.contains(calleeSymbol.fullyQualifiedName());
        }
        return false;
      }
      default -> {
        return false;
      }
    }
  }

  /**
   * Checks that the reference stored in a variable cannot escape.
   */
  private static boolean cannotEscape(Symbol symb) {
    List<Usage> usages = symb.usages();
    if (usages.size() > 2) {
      // Check that all usages are either assignments or `is`-comparisons.
      return usages.stream().allMatch(u -> u.isBindingUsage() ||
        TreeUtils.firstAncestorOfKind(u.tree(), Tree.Kind.IS) != null);
    }
    return true;
  }

}
