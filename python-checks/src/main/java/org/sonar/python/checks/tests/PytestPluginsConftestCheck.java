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
package org.sonar.python.checks.tests;

import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S8999")
public class PytestPluginsConftestCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "\"pytest_plugins\" should be defined in conftest.py files";
  private static final String PYTEST_PLUGINS = "pytest_plugins";
  private static final String CONFTEST_FILE_NAME = "conftest.py";

  @Override
  public CheckScope scope() {
    return CheckScope.TESTS;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, PytestPluginsConftestCheck::onAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, PytestPluginsConftestCheck::onAssignment);
  }

  private static void onAssignment(SubscriptionContext ctx) {
    checkPytestPluginsAssignment(ctx, ctx.syntaxNode());
  }

  private static void checkPytestPluginsAssignment(SubscriptionContext ctx, Tree assignment) {
    if (CONFTEST_FILE_NAME.equals(ctx.pythonFile().fileName()) || !isModuleLevel(assignment)) {
      return;
    }
    getPytestPluginsTarget(assignment).ifPresent(target -> ctx.addIssue(target, MESSAGE));
  }

  private static boolean isModuleLevel(Tree assignment) {
    return TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF) == null;
  }

  private static Optional<Expression> getPytestPluginsTarget(Tree assignment) {
    if (assignment instanceof AssignmentStatement assignmentStatement) {
      return assignmentStatement.lhsExpressions().stream()
        .map(ExpressionList::expressions)
        .flatMap(List::stream)
        .filter(PytestPluginsConftestCheck::isPytestPluginsName)
        .findFirst();
    }
    if (assignment instanceof AnnotatedAssignment annotatedAssignment) {
      Expression lhs = annotatedAssignment.variable();
      if (isPytestPluginsName(lhs) && annotatedAssignment.assignedValue() != null) {
        return Optional.of(lhs);
      }
    }
    return Optional.empty();
  }

  private static boolean isPytestPluginsName(Expression expression) {
    return expression.is(Tree.Kind.NAME) && PYTEST_PLUGINS.equals(((Name) expression).name());
  }
}
