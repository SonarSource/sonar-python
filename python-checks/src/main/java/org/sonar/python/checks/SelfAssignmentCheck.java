/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
import org.sonar.python.PythonBuiltinFunctions;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.PyAliasedNameTree;
import org.sonar.python.api.tree.PyAnnotatedAssignmentTree;
import org.sonar.python.api.tree.PyAssignmentStatementTree;
import org.sonar.python.api.tree.PyCallExpressionTree;
import org.sonar.python.api.tree.PyExpressionTree;
import org.sonar.python.api.tree.PyImportFromTree;
import org.sonar.python.api.tree.PyImportNameTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.tree.BaseTreeVisitor;

@Rule(key = SelfAssignmentCheck.CHECK_KEY)
public class SelfAssignmentCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "S1656";

  public static final String MESSAGE = "Remove or correct this useless self-assignment.";

  private Set<String> importedNames = new HashSet<>();

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> this.importedNames.clear());

    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_FROM, ctx ->
      ((PyImportFromTree) ctx.syntaxNode()).importedNames().forEach(this::addImportedName));

    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_NAME, ctx ->
      ((PyImportNameTree) ctx.syntaxNode()).modules().forEach(this::addImportedName));

    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignement);

    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, this::checkAnnotatedAssignment);
  }

  private void checkAssignement(SubscriptionContext ctx) {
    PyAssignmentStatementTree assignment = (PyAssignmentStatementTree) ctx.syntaxNode();
    PyExpressionTree assignedValue = assignment.assignedValue();
    for (int i = 0; i < assignment.lhsExpressions().size(); i++) {
      List<PyExpressionTree> expressions = assignment.lhsExpressions().get(i).expressions();
      if (expressions.size() == 1 && CheckUtils.areEquivalent(assignedValue, expressions.get(0)) && !isException(assignment, assignedValue)) {
        ctx.addIssue(assignment.equalTokens().get(i), MESSAGE);
      }
    }
  }

  private void checkAnnotatedAssignment(SubscriptionContext ctx) {
    PyAnnotatedAssignmentTree assignment = (PyAnnotatedAssignmentTree) ctx.syntaxNode();
    PyExpressionTree assignedValue = assignment.assignedValue();
    PyExpressionTree variable = assignment.variable();
    if (assignedValue != null && CheckUtils.areEquivalent(assignedValue, variable) && !isException(assignment, assignedValue)) {
      ctx.addIssue(assignment.equalToken(), MESSAGE);
    }
  }

  private void addImportedName(PyAliasedNameTree aliasedName) {
    PyNameTree alias = aliasedName.alias();
    if (alias != null) {
      importedNames.add(alias.name());
    } else {
      List<PyNameTree> names = aliasedName.dottedName().names();
      importedNames.add(names.get(names.size() - 1).name());
    }
  }

  private boolean isException(PyStatementTree assignment, PyExpressionTree assignedValue) {
    if (assignedValue.is(Tree.Kind.NAME) && isAllowedName((PyNameTree) assignedValue)) {
      return true;
    }
    return inClassDef(assignment) || hasCallExpressionDescendant(assignment);
  }

  private boolean isAllowedName(PyNameTree name) {
    return importedNames.contains(name.name()) || PythonBuiltinFunctions.contains(name.name());
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
    public void visitCallExpression(PyCallExpressionTree callExpressionTree) {
      hasCallExpressionDescendant = true;
    }
  }
}
