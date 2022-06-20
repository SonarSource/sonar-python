/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
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
      .forEach(unnecessaryAssignment -> {
        Tree element = unnecessaryAssignment.element;
        String message = String.format(MESSAGE_TEMPLATE, unnecessaryAssignment.symbol.name());
        IssueWithQuickFix issue = (IssueWithQuickFix) separatorTokenForIssue(element)
          .map(separator -> ctx.addIssue(element.firstToken(), separator, message))
          .orElseGet(() -> ctx.addIssue(element, message));

        createQuickFix(issue, element);
      });
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

  private static Optional<Token> separatorTokenForIssue(Tree element) {
    return separatorTokenOfElement(element)
      .filter(sep -> !"\n".equals(sep.value()));
  }

  private static Optional<Token> separatorTokenOfElement(Tree element) {
    if (element.is(Tree.Kind.ASSIGNMENT_STMT, Tree.Kind.EXPRESSION_STMT)) {
      Token separator = ((Statement) element).separator();
      return Optional.ofNullable(separator);
    }
    return Optional.empty();
  }

  private static PythonTextEdit removeDeadStore(Tree currentTree) {
    Optional<Token> tokenSeparator = separatorTokenOfElement(currentTree);
    List<Tree> childrenOfParent = currentTree.parent().children();

    if (childrenOfParent.size() == 1) {
      return PythonTextEdit.replace(currentTree, "pass");
    }

    Token currentFirstToken = currentTree.firstToken();
    int indexOfCurrentTree = childrenOfParent.indexOf(currentTree);

    // If the tree to remove is the last one, we remove from the end of the previous tree until the end of the current one
    if (indexOfCurrentTree == childrenOfParent.size() - 1) {
      Tree previousTree = childrenOfParent.get(indexOfCurrentTree - 1);
      Optional<Token> previousSep = separatorTokenOfElement(previousTree);
      Token previous = previousSep.orElse(previousTree.lastToken());
      currentFirstToken = tokenSeparator.orElse(currentTree.lastToken());
      return removeFromEndOfTillEndOf(previous, currentFirstToken);
    }

    // If the next tree is more than 1 line after the currentFirstToken tree, we only remove the separator
    Token next;
    int currentLine = currentTree.lastToken().line();
    int nextLine = childrenOfParent.get(indexOfCurrentTree + 1).firstToken().line();
    if (nextLine > currentLine + 1) {
      next = tokenSeparator.orElse(currentTree.lastToken());
      // Remove from the start of the currentFirstToken token until after the separator
      return new PythonTextEdit("", currentFirstToken.line(), currentFirstToken.column(), next.line(), next.column() + next.value().length());
    } else {
      // Remove from the start of the currentFirstToken token until the next token
      next = childrenOfParent.get(indexOfCurrentTree + 1).firstToken();
      return new PythonTextEdit("", currentFirstToken.line(), currentFirstToken.column(), next.line(), next.column());
    }
  }

  private static void createQuickFix(IssueWithQuickFix issue, Tree unnecessaryAssignment) {
    PythonTextEdit edit = removeDeadStore(unnecessaryAssignment);
    PythonQuickFix quickFix = PythonQuickFix.newQuickFix("Remove the unused statement")
      .addTextEdit(edit)
      .build();
    issue.addQuickFix(quickFix);
  }

  private static PythonTextEdit removeFromEndOfTillEndOf(Token first, Token last) {
    return new PythonTextEdit("", first.line(), first.column() + first.value().length(),
      last.line(), last.column() + last.value().length());
  }
}
