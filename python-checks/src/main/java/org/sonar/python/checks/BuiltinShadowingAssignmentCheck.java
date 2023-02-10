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

import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.TypeShed;

@Rule(key = "S5806")
public class BuiltinShadowingAssignmentCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Rename this variable; it shadows a builtin.";
  public static final String REPEATED_VAR_MESSAGE = "Variable also assigned here.";
  public static final String QUICK_FIX_MESSAGE_FORMAT = "Rename to _%s";
  private final Map<Symbol, PreciseIssue> variableIssuesRaised = new HashMap<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> variableIssuesRaised.clear());
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, this::checkAnnotatedAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_EXPRESSION, this::checkAssignmentExpression);
  }

  private void checkAssignmentExpression(SubscriptionContext ctx) {
    AssignmentExpression assignmentExpression = (AssignmentExpression) ctx.syntaxNode();
    Name lhsName = assignmentExpression.lhsName();
    if (shouldReportIssue(lhsName)) {
      raiseIssueForNonGlobalVariable(ctx, lhsName);
    }
  }

  private void checkAssignment(SubscriptionContext ctx) {
    AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();
    Tree ancestor = TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF);
    if (ancestor == null || ancestor.is(Tree.Kind.FUNCDEF)) {
      for (int i = 0; i < assignment.lhsExpressions().size(); i++) {
        for (Expression expression : assignment.lhsExpressions().get(i).expressions()) {
          if (shouldReportIssue(expression)) {
            raiseIssueForNonGlobalVariable(ctx, (Name) expression);
          }
        }
      }
    }
  }

  private void checkAnnotatedAssignment(SubscriptionContext ctx) {
    AnnotatedAssignment assignment = (AnnotatedAssignment) ctx.syntaxNode();
    Tree ancestor = TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF);
    if (ancestor == null || ancestor.is(Tree.Kind.FUNCDEF)) {
      Expression variable = assignment.variable();
      Token equalToken = assignment.equalToken();
      if (equalToken != null && shouldReportIssue(variable)) {
        raiseIssueForNonGlobalVariable(ctx, (Name) variable);
      }
    }
  }

  private void raiseIssueForNonGlobalVariable(SubscriptionContext ctx, Name variable) {
    Optional.ofNullable(variable.symbol())
      .filter(symbol -> symbol.usages().stream().map(Usage::kind).noneMatch(Usage.Kind.GLOBAL_DECLARATION::equals))
      .ifPresent(symbol -> {
        var existingIssue = variableIssuesRaised.get(symbol);
        if (existingIssue != null) {
          existingIssue.secondary(variable, REPEATED_VAR_MESSAGE);
        } else {
          var issue = (IssueWithQuickFix) ctx.addIssue(variable, MESSAGE);

          PythonQuickFix quickFix = createQuickFix(symbol);
          issue.addQuickFix(quickFix);
          variableIssuesRaised.put(symbol, issue);
        }
      });
  }

  private static PythonQuickFix createQuickFix(Symbol symbol) {
    var edits = symbol.usages()
      .stream()
      .map(Usage::tree)
      .map(Tree::firstToken)
      .sorted(Comparator.comparing(Token::line).reversed().thenComparing(Token::column).reversed())
      .map(token -> PythonTextEdit.insertBefore(token, "_"))
      .collect(Collectors.toList());


    return PythonQuickFix.newQuickFix(String.format(QUICK_FIX_MESSAGE_FORMAT, symbol.name()))
      .addTextEdit(edits)
      .build();
  }

  private boolean shouldReportIssue(Tree tree) {
    return tree.is(Tree.Kind.NAME) && isBuiltInName((Name) tree) && TreeUtils.firstAncestorOfKind(tree.parent(), Tree.Kind.FUNCDEF) != null;
  }

  private boolean isBuiltInName(Name name) {
    return TypeShed.builtinSymbols().containsKey(name.name());
  }
}
