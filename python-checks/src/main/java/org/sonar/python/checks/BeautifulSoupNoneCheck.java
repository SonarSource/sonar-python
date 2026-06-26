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

import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.AssertStatement;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.TryStatement;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.IsExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8904")
public class BeautifulSoupNoneCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Check if this element exists before accessing it with `%s`.";

  private static TypeMatcher bs4Matcher(String suffix) {
    return TypeMatchers.isType("bs4.element." + suffix);
  }

  // Covers find/select_one (defined on Tag) and the find_* / find_parent methods (defined on PageElement)
  private static final TypeMatcher IS_BS4_SEARCH_CALL = TypeMatchers.any(
    bs4Matcher("Tag.find"),
    bs4Matcher("Tag.select_one"),
    bs4Matcher("PageElement.find_next"),
    bs4Matcher("PageElement.find_previous"),
    bs4Matcher("PageElement.find_next_sibling"),
    bs4Matcher("PageElement.find_previous_sibling"),
    bs4Matcher("PageElement.find_parent"));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.QUALIFIED_EXPR, BeautifulSoupNoneCheck::checkQualifiedExpr);
    context.registerSyntaxNodeConsumer(Tree.Kind.SUBSCRIPTION, BeautifulSoupNoneCheck::checkSubscription);
  }

  private static void checkQualifiedExpr(SubscriptionContext ctx) {
    QualifiedExpression qe = (QualifiedExpression) ctx.syntaxNode();
    // When a QualifiedExpression is the callee of a BS4 search call (e.g. "soup.find"),
    // skip it — unless its qualifier is itself a BS4 search call result (e.g. "soup.find("div").find").
    // The former is safe (soup is not None); the latter is unsafe (soup.find("div") may be None).
    if (qe.parent() instanceof CallExpression callParent
      && callParent.callee() == qe
      && IS_BS4_SEARCH_CALL.isTrueFor(qe, ctx)
      && !(qe.qualifier() instanceof CallExpression qualifierCall && isBS4SearchCall(qualifierCall, ctx))) {
      return;
    }
    checkObjectAccess(qe.qualifier(), qe.name(), "." + qe.name().name(), ctx);
  }

  private static void checkSubscription(SubscriptionContext ctx) {
    SubscriptionExpression se = (SubscriptionExpression) ctx.syntaxNode();
    if (isAssignmentTarget(se)) {
      return;
    }
    checkObjectAccess(se.object(), se, subscriptDescription(se), ctx);
  }

  private static String subscriptDescription(SubscriptionExpression se) {
    var subscripts = se.subscripts().expressions();
    if (subscripts.size() == 1 && subscripts.get(0) instanceof StringLiteral str) {
      return String.format("[%s]", str.trimmedQuotesValue());
    }
    return "[]";
  }

  private static void checkObjectAccess(Expression object, Tree issueLocation, String accessDescription, SubscriptionContext ctx) {
    if (isInsideCatchingTry(issueLocation, ctx)) {
      return;
    }
    // Case 1: direct inline chaining — the object is itself a BS4 search call.
    // Only raise on the first unsafe access in a chain: skip if the object's own qualifier is
    // already a BS4 search call (meaning an earlier issue was already raised for that access).
    // e.g. soup.find("div").find("p").text — raise on the first find, not the second
    if (object instanceof CallExpression callExpr && isBS4SearchCall(callExpr, ctx)) {
      if (callExpr.callee() instanceof QualifiedExpression innerQe
        && innerQe.qualifier() instanceof CallExpression innerQualifierCall
        && isBS4SearchCall(innerQualifierCall, ctx)) {
        return;
      }
      // Highlight the name of the call that produced the potentially-None object
      // e.g. for soup.find("p").text → highlight "find"; for soup.find("div").find("p") → highlight the first "find"
      Tree calleeNameLocation = callExpr.callee() instanceof QualifiedExpression calleeQe
        ? calleeQe.name()
        : issueLocation;
      ctx.addIssue(calleeNameLocation, MESSAGE.formatted(accessDescription));
      return;
    }
    // Case 2: variable whose value at this location comes from a BS4 search call
    if (object instanceof Name name) {
      checkNameAccess(name, issueLocation, accessDescription, ctx);
    }
  }

  private static boolean isInsideCatchingTry(Tree tree, SubscriptionContext ctx) {
    TryStatement enclosingTry = (TryStatement) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.TRY_STMT);
    return enclosingTry != null
      && TreeUtils.hasDescendant(enclosingTry.body(), t -> t == tree)
      && catchesAttributeError(enclosingTry.exceptClauses(), ctx);
  }

  private static void checkNameAccess(Name name, Tree issueLocation, String accessDescription, SubscriptionContext ctx) {
    Set<Expression> values = ctx.valuesAtLocation(name);
    if (values.isEmpty()) {
      return;
    }
    boolean allBS4Calls = values.stream()
      .allMatch(v -> v instanceof CallExpression callExpr && isBS4SearchCall(callExpr, ctx));
    if (!allBS4Calls) {
      return;
    }
    if (isGuardedByNoneCheck(name)) {
      return;
    }
    ctx.addIssue(issueLocation, MESSAGE.formatted(accessDescription));
  }

  private static boolean isBS4SearchCall(CallExpression callExpr, SubscriptionContext ctx) {
    return IS_BS4_SEARCH_CALL.isTrueFor(callExpr.callee(), ctx);
  }

  /**
   * Returns true if the Name is guarded against None. Recognises three patterns:
   * 1. Enclosing if-body guard: walks *all* enclosing IfStatements (not just the nearest) so that
   *    nested ifs inside a guard are covered. Conditions may include "and"-chains.
   *    e.g. "if name:", "if name is not None:", "if name != None:", "if name is not None and other:"
   * 2. Early-exit guard: a preceding usage of the same symbol appears as the subject of an
   *    IfStatement that exits unconditionally (return/raise/continue/break) when the name is
   *    None or falsy: "if name is None: return" / "if not name: return"
   * 3. Assert guard: a preceding assert statement whose condition matches conditionChecksForTruthy:
   *    "assert name" / "assert name is not None"
   */
  private static boolean isGuardedByNoneCheck(Name name) {
    // Pattern 1: walk all enclosing IfStatements, not just the nearest
    Tree cursor = name.parent();
    while (cursor != null) {
      if (cursor instanceof IfStatement ifStmt
        && TreeUtils.hasDescendant(ifStmt.body(), t -> t == name)
        && conditionChecksForTruthy(ifStmt.condition(), name)) {
        return true;
      }
      cursor = cursor.parent();
    }

    SymbolV2 symbol = name.symbolV2();
    if (symbol == null) {
      return false;
    }
    int nameLine = name.firstToken().line();

    // Pattern 2: an earlier usage of the same symbol is guarded by an early-exit if-statement
    if (symbol.usages().stream()
      .filter(u -> u.tree().firstToken().line() < nameLine)
      .map(u -> TreeUtils.firstAncestorOfKind(u.tree(), Tree.Kind.IF_STMT))
      .filter(IfStatement.class::isInstance)
      .map(IfStatement.class::cast)
      .anyMatch(ifStmt -> isEarlyExitNoneGuard(ifStmt, name))) {
      return true;
    }

    // Pattern 3: assert statement whose condition guards the name
    return symbol.usages().stream()
      .filter(u -> u.tree().firstToken().line() < nameLine)
      .map(u -> TreeUtils.firstAncestorOfKind(u.tree(), Tree.Kind.ASSERT_STMT))
      .filter(AssertStatement.class::isInstance)
      .map(AssertStatement.class::cast)
      .anyMatch(assertStmt -> conditionChecksForTruthy(assertStmt.condition(), name));
  }

  private static final TypeMatcher IS_ATTRIBUTE_ERROR =
    TypeMatchers.isOrExtendsType("builtins.AttributeError");

  private static boolean catchesAttributeError(List<ExceptClause> exceptClauses, SubscriptionContext ctx) {
    for (ExceptClause clause : exceptClauses) {
      Expression exception = clause.exception();
      if (exception == null) {
        // bare "except:" — too broad, do not suppress
        return false;
      }
      // check each exception type in the clause (handles "except (AttributeError, TypeError):")
      List<Expression> caught = TreeUtils.flattenTuples(exception).toList();
      for (Expression expr : caught) {
        if (IS_ATTRIBUTE_ERROR.isTrueFor(expr, ctx)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Returns true if the if-statement exits unconditionally (return/raise/continue/break) and
   * its condition tests the name for None or falsiness.
   * Recognised conditions: "name is None", "not name", "name == None"
   */
  private static boolean isEarlyExitNoneGuard(IfStatement ifStmt, Name name) {
    var stmts = ifStmt.body().statements();
    if (stmts.isEmpty()) {
      return false;
    }
    if (!stmts.get(stmts.size() - 1).is(Tree.Kind.RETURN_STMT, Tree.Kind.RAISE_STMT, Tree.Kind.CONTINUE_STMT, Tree.Kind.BREAK_STMT)) {
      return false;
    }
    return conditionChecksForNoneOrFalsy(ifStmt.condition(), name);
  }

  /**
   * Checks if the condition tests the name for None or falsiness (the inverse of a positive guard).
   * Recognised: "not name", "name is None", "name == None"
   */
  private static boolean conditionChecksForNoneOrFalsy(Expression condition, Name name) {
    // "if not element:"
    if (condition instanceof UnaryExpression unary
      && "not".equals(unary.operator().value())
      && nameMatchesExpressionName(name, unary.expression())) {
      return true;
    }
    // "if element is None:" / "if None is element:"
    if (condition instanceof IsExpression isExpr && isExpr.notToken() == null) {
      return nameComparedToNone(name, isExpr.leftOperand(), isExpr.rightOperand());
    }
    // "if element == None:" / "if None == element:"
    if (condition instanceof BinaryExpression binaryExpr
      && condition.is(Tree.Kind.COMPARISON)
      && "==".equals(binaryExpr.operator().value())) {
      return nameComparedToNone(name, binaryExpr.leftOperand(), binaryExpr.rightOperand());
    }
    return false;
  }

  private static boolean conditionChecksForTruthy(Expression condition, Name name) {
    // "if element:"
    if (nameMatchesExpressionName(name, condition)) {
      return true;
    }
    // "if <guard> and other:" / "if other and <guard>:" — recurse into both operands of "and"
    if (condition instanceof BinaryExpression andExpr
      && condition.is(Tree.Kind.AND)
      && (conditionChecksForTruthy(andExpr.leftOperand(), name)
        || conditionChecksForTruthy(andExpr.rightOperand(), name))) {
      return true;
    }
    // "if element is not None:" / "if None is not element:"
    if (condition instanceof IsExpression isExpr && isExpr.notToken() != null) {
      return nameComparedToNone(name, isExpr.leftOperand(), isExpr.rightOperand());
    }
    // "if element != None:" / "if None != element:"
    if (condition instanceof BinaryExpression binaryExpr
      && condition.is(Tree.Kind.COMPARISON)
      && "!=".equals(binaryExpr.operator().value())) {
      return nameComparedToNone(name, binaryExpr.leftOperand(), binaryExpr.rightOperand());
    }
    return false;
  }

  /** Returns true if one operand is the name and the other is None (in either order). */
  private static boolean nameComparedToNone(Name name, Expression left, Expression right) {
    return (nameMatchesExpressionName(name, left) && right.is(Tree.Kind.NONE))
      || (nameMatchesExpressionName(name, right) && left.is(Tree.Kind.NONE));
  }

  private static boolean nameMatchesExpressionName(Name name, Expression operand) {
    return operand instanceof Name opName && opName.name().equals(name.name());
  }

  private static boolean isAssignmentTarget(SubscriptionExpression subscriptionExpression) {
    return TreeUtils.firstAncestor(subscriptionExpression,
      t -> t.is(Tree.Kind.ASSIGNMENT_STMT)
        && ((AssignmentStatement) t).lhsExpressions().stream()
          .flatMap(lhs -> lhs.expressions().stream())
          .anyMatch(expr -> expr.equals(subscriptionExpression))) != null;
  }
}
