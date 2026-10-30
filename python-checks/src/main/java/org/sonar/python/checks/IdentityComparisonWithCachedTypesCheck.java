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

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.symbols.v2.UsageV2;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.checks.utils.ReachingValues;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;


// https://jira.sonarsource.com/browse/RSPEC-5795
@Rule(key = "S5795")
public class IdentityComparisonWithCachedTypesCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE_IS = "Replace this \"is\" operator with \"==\"; identity operator is not reliable here.";
  private static final String MESSAGE_IS_NOT = "Replace this \"is not\" operator with \"!=\"; identity operator is not reliable here.";
  public static final String IS_QUICK_FIX_MESSAGE = "Replace with \"==\"";
  public static final String IS_NOT_QUICK_FIX_MESSAGE = "Replace with \"!=\"";

  private static final TypeMatcher ENUM_CLASS = TypeMatchers.isOrExtendsType("enum.Enum");

  private static final TypeMatcher BUILTIN_FACTORY = TypeMatchers.any(
    Stream.of("frozenset", "bytes", "int", "float", "str", "tuple", "hash")
      .map(name -> TypeMatchers.isType("builtins." + name)));
  private static final TypeMatcher UNSUITABLE_TYPE = TypeMatchers.any(
    Stream.of("frozenset", "bytes", "int", "float", "tuple")
      .map(name -> TypeMatchers.isObjectOfType("builtins." + name)));
  // NoneType is a synthetic builtin exposed at the root of the V2 type table.
  private static final TypeMatcher NONE_TYPE = TypeMatchers.any(
    TypeMatchers.isType("NoneType"), TypeMatchers.isObjectOfType("NoneType"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.IS, IdentityComparisonWithCachedTypesCheck::checkIsComparison);
  }

  private static void checkIsComparison(SubscriptionContext ctx) {
    IsExpression isExpr = (IsExpression) ctx.syntaxNode();

    // Comparison to none is checked by S5727
    if (isComparisonToNone(isExpr, ctx)) {
      return;
    }

    if (isUnsuitableOperand(isExpr.leftOperand(), ctx) || isUnsuitableOperand(isExpr.rightOperand(), ctx)) {
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

  private static boolean isComparisonToNone(IsExpression isExpr, SubscriptionContext ctx) {
    return NONE_TYPE.isTrueFor(isExpr.leftOperand(), ctx) || NONE_TYPE.isTrueFor(isExpr.rightOperand(), ctx);
  }

  /**
   * Checks whether an expression is either of a type that should generally be compared with <code>==</code>, or
   * whether it holds a non-escaping reference that stems from a function that guarantees that the
   * returned reference is not shared anywhere.
   */
  private static boolean isUnsuitableOperand(Expression expr, SubscriptionContext ctx) {
    if (isEnumMember(expr, ctx)) {
      return false;
    }
    if (UNSUITABLE_TYPE.isTrueFor(expr, ctx)) {
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
      return BUILTIN_FACTORY.isTrueFor(callExpr.callee(), ctx);
    }
    return false;
  }

  private static boolean isEnumMember(Expression expression, SubscriptionContext ctx) {
    Set<Expression> visited = new HashSet<>();
    Expression value = Expressions.removeParentheses(expression);
    while (value instanceof Name name && visited.add(value)) {
      value = ReachingValues.singleValue(name, ctx);
      if (value == null) {
        return false;
      }
      value = Expressions.removeParentheses(value);
    }
    // Enum class members may retain their scalar initializer type in stubs.
    // Their runtime values are Enum singletons, not those scalar initializers.
    return value instanceof QualifiedExpression member && ENUM_CLASS.isTrueFor(member.qualifier(), ctx);
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
