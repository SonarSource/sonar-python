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
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParenthesizedExpression;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.semantic.BuiltinSymbols;
import org.sonar.python.tree.TreeUtils;

@Rule(key = SelfAssignmentCheck.CHECK_KEY)
public class SelfAssignmentCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "S1656";

  public static final String MESSAGE = "Remove or correct this useless self-assignment.";
  private static final String QUICK_FIX_MESSAGE = "Remove the self-assignment";

  private Set<String> importedNames = new HashSet<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> this.importedNames.clear());

    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_FROM, ctx ->
      ((ImportFrom) ctx.syntaxNode()).importedNames().forEach(this::addImportedName));

    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_NAME, ctx ->
      ((ImportName) ctx.syntaxNode()).modules().forEach(this::addImportedName));

    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignment);

    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, this::checkAnnotatedAssignment);

    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_EXPRESSION, SelfAssignmentCheck::checkAssignmentExpression);
  }

  private static void checkAssignmentExpression(SubscriptionContext ctx) {
    AssignmentExpression assignmentExpression = (AssignmentExpression) ctx.syntaxNode();
    if (CheckUtils.areEquivalent(assignmentExpression.lhsName(), assignmentExpression.expression())) {
      var issue = ctx.addIssue(assignmentExpression.operator(), MESSAGE);
      issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE, createAssignmentExpressionQuickFix(assignmentExpression)));
    }
  }

  private static PythonTextEdit createAssignmentExpressionQuickFix(AssignmentExpression assignmentExpression) {
    Expression expression = assignmentExpression.expression();
    Tree parent = assignmentExpression.parent();
    String replacement = TreeUtils.treeToString(expression, false);
    if (parent instanceof ParenthesizedExpression) {
      return TextEditUtils.replace(parent, replacement);
    }
    return TextEditUtils.replace(assignmentExpression, replacement);
  }

  private void checkAssignment(SubscriptionContext ctx) {
    AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();
    Expression assignedValue = assignment.assignedValue();
    for (int i = 0; i < assignment.lhsExpressions().size(); i++) {
      ExpressionList expressionList = assignment.lhsExpressions().get(i);
      if (expressionList.commas().isEmpty() && CheckUtils.areEquivalent(assignedValue, expressionList.expressions().get(0)) && !isException(assignment, assignedValue)) {
        var issue = ctx.addIssue(assignment.equalTokens().get(i), MESSAGE);
        issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE, TextEditUtils.removeStatement(assignment)));
      }
    }
  }

  private void checkAnnotatedAssignment(SubscriptionContext ctx) {
    AnnotatedAssignment assignment = (AnnotatedAssignment) ctx.syntaxNode();
    Expression assignedValue = assignment.assignedValue();
    Expression variable = assignment.variable();
    if (assignedValue != null && CheckUtils.areEquivalent(assignedValue, variable) && !isException(assignment, assignedValue)) {
      var issue = ctx.addIssue(assignment.equalToken(), MESSAGE);
      issue.addQuickFix(PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE, removeAnnotatedAssignedValue(assignment)));
    }
  }

  private static PythonTextEdit removeAnnotatedAssignedValue(AnnotatedAssignment assignment) {
    var equalToken = assignment.equalToken();
    var assignedValue = assignment.assignedValue();
    if (assignedValue == null) {
      throw new IllegalStateException("Annotated assignment should have an assigned value.");
    }
    return TextEditUtils.removeRange(equalToken.pythonLine(), equalToken.column(), assignedValue.lastToken().pythonLine(),
      assignedValue.lastToken().column() + assignedValue.lastToken().value().length());
  }

  private void addImportedName(AliasedName aliasedName) {
    Name alias = aliasedName.alias();
    if (alias != null) {
      importedNames.add(alias.name());
    } else {
      List<Name> names = aliasedName.dottedName().names();
      importedNames.add(names.get(names.size() - 1).name());
    }
  }

  private boolean isException(Statement assignment, Expression assignedValue) {
    if (assignedValue.is(Tree.Kind.NAME) && isAllowedName((Name) assignedValue)) {
      return true;
    }
    return inClassDef(assignment) || hasCallExpressionDescendant(assignment);
  }

  private boolean isAllowedName(Name name) {
    return importedNames.contains(name.name()) || BuiltinSymbols.all().contains(name.name());
  }

  private static boolean inClassDef(Tree tree) {
    Tree currentParent = tree.parent();
    currentParent = currentParent.is(Tree.Kind.STATEMENT_LIST) ? currentParent.parent() : currentParent;
    return currentParent.is(Tree.Kind.CLASSDEF);
  }

  private static boolean hasCallExpressionDescendant(Tree tree) {
    CallExpressionDescendantVisitor visitor = new CallExpressionDescendantVisitor();
    tree.accept(visitor);
    return visitor.hasCallExpressionDescendant;
  }

  private static class CallExpressionDescendantVisitor extends BaseTreeVisitor {
    private boolean hasCallExpressionDescendant = false;

    @Override
    public void visitCallExpression(CallExpression callExpressionTree) {
      hasCallExpressionDescendant = true;
    }
  }
}
