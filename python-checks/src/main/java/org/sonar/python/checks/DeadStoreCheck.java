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

import java.util.Set;
import java.util.stream.IntStream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.cfg.fixpoint.LiveVariablesAnalysis;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.DeadStoreUtils.isUsedInSubFunction;

@Rule(key = "S1854")
public class DeadStoreCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_TEMPLATE = "Remove this assignment to local variable '%s'; the value is never used.";

  private static final String SECONDARY_MESSAGE_TEMPLATE = "'%s' is reassigned here.";
  public static final String QUICK_FIX_MESSAGE = "Remove the unused assignment";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (TreeUtils.hasDescendant(functionDef, tree -> tree.is(Tree.Kind.TRY_STMT))) {
        return;
      }
      ControlFlowGraph cfg = ControlFlowGraph.build(functionDef, ctx.pythonFile());
      if (cfg == null) {
        return;
      }
      LiveVariablesAnalysis lva = LiveVariablesAnalysis.analyze(cfg);
      cfg.blocks().forEach(block -> verifyBlock(ctx, block, lva.getLiveVariables(block), lva.getReadSymbols(), functionDef));
    });
  }

  /**
   * Bottom-up approach, keeping track of which variables will be read by successor elements.
   */
  private static void verifyBlock(SubscriptionContext ctx, CfgBlock block, LiveVariablesAnalysis.LiveVariables blockLiveVariables,
    Set<Symbol> readSymbols, FunctionDef functionDef) {

    DeadStoreUtils.findUnnecessaryAssignments(block, blockLiveVariables, functionDef)
      .stream()
      // symbols should have at least one read usage (otherwise will be reported by S1481)
      .filter(unnecessaryAssignment -> readSymbols.contains(unnecessaryAssignment.symbol))
      .filter((unnecessaryAssignment -> !isException(unnecessaryAssignment.symbol, unnecessaryAssignment.element, functionDef)))
      .forEach(unnecessaryAssignment -> raiseIssue(ctx, unnecessaryAssignment));
  }

  private static void raiseIssue(SubscriptionContext ctx, DeadStoreUtils.UnnecessaryAssignment unnecessaryAssignment) {
    Tree element = unnecessaryAssignment.element;
    String symbolName = unnecessaryAssignment.symbol.name();
    String message = String.format(MESSAGE_TEMPLATE, symbolName);
    Token lastRelevantToken = TreeUtils.getTreeSeparatorOrLastToken(element);
    IssueWithQuickFix issue;
    if ("\n".equals(lastRelevantToken.value())) {
      issue = (IssueWithQuickFix) ctx.addIssue(element, message);
    } else {
      issue = (IssueWithQuickFix) ctx.addIssue(element.firstToken(), lastRelevantToken, message);
    }

    unnecessaryAssignment.symbol.usages().stream()
      .filter(Usage::isBindingUsage)
      .map(Usage::tree)
      // skip initial issue binding
      .filter(tree -> tree != element && TreeUtils.firstAncestor(tree, parent -> parent == element) == null)
      //skip assignments before
      .filter(tree -> TreeUtils.getTreeByPositionComparator().compare(tree, element) > 0)
      .collect(TreeUtils.groupAssignmentByParentStatementList())
      .values()
      .stream()
      .sorted(TreeUtils.getTreeByPositionComparator())
      .map(DeadStoreCheck::mapToParentAssignmentStatementOrExpression)
      .forEach(tree -> issue.secondary(tree, String.format(SECONDARY_MESSAGE_TEMPLATE, symbolName)));

    if (element instanceof Statement && !isExceptionForQuickFix((Statement) element)) {
      if (element.is(Tree.Kind.ASSIGNMENT_STMT) && ((AssignmentStatement) element).lhsExpressions().size() > 1) {
        addMultipleAssignmentStatementQuickFix((AssignmentStatement) element, issue, unnecessaryAssignment.symbol);
      } else {
        issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE,
          PythonTextEdit.removeStatement((Statement) element)));
      }
    }

  }

  private static void addMultipleAssignmentStatementQuickFix(AssignmentStatement element, IssueWithQuickFix issue, Symbol symbol) {
    var children = element.children();

    IntStream.range(0, children.size())
      .filter(i -> isExpressionHasSymbol(children.get(i), symbol))
      .findFirst()
      .ifPresent(i -> {
        var from = i == 0 ? i : (i - 1);
        var to = from + 2;

        issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE,
          PythonTextEdit.removeUntil(children.get(from), children.get(to))));
      });
  }

  private static boolean isExpressionHasSymbol(Tree element, Symbol symbol) {
    return element.children()
      .stream()
      .filter(HasSymbol.class::isInstance)
      .map(HasSymbol.class::cast)
      .map(HasSymbol::symbol)
      .anyMatch(s -> s == symbol);
  }

  private static Tree mapToParentAssignmentStatementOrExpression(Tree tree) {
    var assignment = TreeUtils.firstAncestor(tree, parent -> parent.is(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ASSIGNMENT_EXPRESSION));
    if (assignment != null) {
      return assignment;
    }
    return tree;
  }

  private static boolean isMultipleAssignement(Tree element) {
    return element.is(Tree.Kind.ASSIGNMENT_STMT) &&
      ((AssignmentStatement) element).lhsExpressions().stream().anyMatch(lhsExpression -> lhsExpression.expressions().size() > 1);
  }

  private static boolean isException(Symbol symbol, Tree element, FunctionDef functionDef) {
    return isUnderscoreVariable(symbol)
      || isAssignmentToFalsyOrTrueLiteral(element)
      || isFunctionDeclarationSymbol(symbol)
      || isLoopDeclarationSymbol(symbol, element)
      || isWithInstance(element)
      || isUsedInSubFunction(symbol, functionDef)
      || DeadStoreUtils.isParameter(element)
      || isMultipleAssignement(element)
      || isAnnotatedAssignmentWithoutRhs(element);
  }

  private static boolean isAnnotatedAssignmentWithoutRhs(Tree element) {
    return element.is(Tree.Kind.ANNOTATED_ASSIGNMENT) && ((AnnotatedAssignment) element).assignedValue() == null;
  }

  private static boolean isUnderscoreVariable(Symbol symbol) {
    return symbol.name().equals("_");
  }

  private static boolean isLoopDeclarationSymbol(Symbol symbol, Tree element) {
    return symbol.usages().stream().anyMatch(u -> u.kind() == Usage.Kind.LOOP_DECLARATION)
      && TreeUtils.firstAncestorOfKind(element, Tree.Kind.FOR_STMT) != null;
  }

  private static boolean isWithInstance(Tree element) {
    return element.is(Tree.Kind.WITH_ITEM);
  }

  private static boolean isAssignmentToFalsyOrTrueLiteral(Tree element) {
    if (element.is(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.ANNOTATED_ASSIGNMENT)) {
      Expression assignedValue = element.is(Tree.Kind.ASSIGNMENT_STMT)
        ? ((AssignmentStatement) element).assignedValue()
        : ((AnnotatedAssignment) element).assignedValue();
      return assignedValue != null && (Expressions.isFalsy(assignedValue)
        || (assignedValue.is(Tree.Kind.NAME) && "True".equals(((Name) assignedValue).name()))
        || isNumericLiteralOne(assignedValue)
        || isMinusOne(assignedValue));
    }
    return false;
  }

  private static boolean isMinusOne(Expression assignedValue) {
    if (assignedValue.is(Tree.Kind.UNARY_MINUS)) {
      Expression expression = ((UnaryExpression) assignedValue).expression();
      return isNumericLiteralOne(expression);
    }
    return false;
  }

  private static boolean isNumericLiteralOne(Expression expression) {
    return (expression.is(Tree.Kind.NUMERIC_LITERAL) && "1".equals((((NumericLiteral) expression).valueAsString())));
  }

  private static boolean isFunctionDeclarationSymbol(Symbol symbol) {
    return symbol.usages().stream().anyMatch(u -> u.kind() == Usage.Kind.FUNC_DECLARATION);
  }


  private static boolean isExceptionForQuickFix(Statement tree) {
    switch (tree.getKind()) {
      // foo:str = bar
      case ANNOTATED_ASSIGNMENT:
        return SideEffectDetector.hasSideEffect(((AnnotatedAssignment) tree).assignedValue());
      // foo = bar or foo = bar = 1
      case ASSIGNMENT_STMT:
        return SideEffectDetector.hasSideEffect(((AssignmentStatement) tree).assignedValue());
      // foo(bar:=3)
      case EXPRESSION_STMT:
        ExpressionStatement expressionStatement = (ExpressionStatement) tree;
        return expressionStatement.expressions().stream().anyMatch(SideEffectDetector::hasSideEffect);
      default:
        return false;
    }
  }

  private static class SideEffectDetector extends BaseTreeVisitor {

    private boolean sideEffect = false;

    public static boolean hasSideEffect(@Nullable Expression expression) {
      if (expression == null) {
        return false;
      }
      SideEffectDetector detector = new SideEffectDetector();
      detector.scan(expression);
      return detector.sideEffect;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      sideEffect = true;
    }

  }
}
