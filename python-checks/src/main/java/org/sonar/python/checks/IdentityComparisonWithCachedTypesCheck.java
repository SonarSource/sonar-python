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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeChecker;

import static java.util.Arrays.asList;

// https://jira.sonarsource.com/browse/RSPEC-5795
// https://jira.sonarsource.com/browse/SONARPY-673
@Rule(key = "S5795")
public class IdentityComparisonWithCachedTypesCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE_IS = "Replace this \"is\" operator with \"==\"; identity operator is not reliable here.";
  private static final String MESSAGE_IS_NOT = "Replace this \"is not\" operator with \"!=\"; identity operator is not reliable here.";
  public static final String IS_QUICK_FIX_MESSAGE = "Replace with \"==\"";
  public static final String IS_NOT_QUICK_FIX_MESSAGE = "Replace with \"!=\"";

  /**
   * Fully qualified names of constructors and functions that are guaranteed to create fresh objects with
   * references not shared anywhere else.
   * <p>
   * If a reference arises from a call to one of these functions, and it does not escape anywhere else, then an
   * <code>is</code>-comparison with such a reference will always return <code>False</code>.
   * <p>
   * Note that these are fully qualified names of (value-level) expressions, not types.
   */
  private static final List<String> FQNS_CONSTRUCTORS_RETURNING_UNIQUE_REF =
    asList("frozenset", "bytes", "int", "float", "str", "tuple", "hash");

  /**
   * Names of types that usually should not be compared with <code>is</code>.
   * <p>
   * Note that these are names of types, not `fqn`s of expressions.
   */
  private static final List<String> NAMES_OF_TYPES_UNSUITABLE_FOR_COMPARISON =
    asList("frozenset", "bytes", "int", "float", "tuple");

  private TypeChecker typeChecker;
  private TypeCheckBuilder isNoneTypeChecker;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      typeChecker = ctx.typeChecker();
      isNoneTypeChecker = ctx.typeChecker().typeCheckBuilder().isBuiltinWithName(BuiltinTypes.NONE_TYPE);
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.IS, this::checkIsComparison);
  }

  private void checkIsComparison(SubscriptionContext ctx) {
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

  private boolean isComparisonToNone(IsExpression isExpr) {
    return isNone(isExpr.leftOperand().typeV2()) || isNone(isExpr.rightOperand().typeV2());
  }

  private boolean isNone(PythonType type) {
    return isNoneTypeChecker.check(type) == TriBool.TRUE;
  }

  /**
   * Checks whether an expression is either of a type that should generally be compared with <code>==</code>, or
   * whether it holds a non-escaping reference that stems from a function that guarantees that the
   * returned reference is not shared anywhere.
   */
  private boolean isUnsuitableOperand(Expression expr) {
    PythonType type = expr.typeV2();
    if (isUnsuitableType(type)) {
      if (expr instanceof Name name) {
        SymbolV2 symbol = name.symbolV2();
        return symbol != null && !isVariableThatCanEscape(symbol);
      } else {
        return true;
      }
    }
    if (expr.is(Tree.Kind.STRING_LITERAL)) {
      return true;
    } else if (expr instanceof CallExpression callExpr) {
      PythonType calleeType = callExpr.callee().typeV2();
      return isConstructorReturningUniqueRef(calleeType);
    }
    return false;
  }

  private boolean isUnsuitableType(PythonType type) {
    for (String builtinName : NAMES_OF_TYPES_UNSUITABLE_FOR_COMPARISON) {
      TypeCheckBuilder builtinWithNameChecker = typeChecker.typeCheckBuilder().isBuiltinWithName(builtinName);
      if (builtinWithNameChecker.check(type) == TriBool.TRUE) {
        return true;
      }
    }
    return false;
  }

  private boolean isConstructorReturningUniqueRef(PythonType type) {
    for (String constructorName : FQNS_CONSTRUCTORS_RETURNING_UNIQUE_REF) {
      TypeCheckBuilder constructorChecker = typeChecker.typeCheckBuilder().isTypeWithName(constructorName);
      if (constructorChecker.check(type) == TriBool.TRUE) {
        return true;
      }
    }
    return false;
  }

  /**
   * Checks whether a variable is used anywhere except at the <code>is</code>-comparison site and in the defining
   * assignment. In such cases, we say that the reference to the value of the variable could have escaped,
   * and don't raise an issue. We do not check whether it could actually arrive in the other operand.
   */
  private static boolean isVariableThatCanEscape(SymbolV2 symb) {
    List<UsageV2> usages = symb.usages();
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
