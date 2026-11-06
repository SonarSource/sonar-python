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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CompoundAssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S9459")
public class FrozendictAssignmentCheck extends PythonSubscriptionCheck {
  private static final String MESSAGE = "Remove this assignment to an item or attribute of a frozendict.";
  private static final TypeMatcher FROZENDICT = TypeMatchers.isObjectOfType("builtins.frozendict");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, FrozendictAssignmentCheck::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.COMPOUND_ASSIGNMENT, FrozendictAssignmentCheck::checkCompoundAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, FrozendictAssignmentCheck::checkAnnotatedAssignment);
  }

  private static void checkAssignment(SubscriptionContext context) {
    AssignmentStatement assignment = (AssignmentStatement) context.syntaxNode();
    assignment.lhsExpressions().stream()
      .flatMap(expressions -> expressions.expressions().stream())
      .forEach(expression -> checkAssignmentTarget(context, expression));
  }

  private static void checkAssignmentTarget(SubscriptionContext context, Expression expression) {
    if (expression instanceof Tuple tuple) {
      tuple.elements().forEach(element -> checkAssignmentTarget(context, element));
      return;
    }
    if (expression instanceof ListLiteral listLiteral) {
      listLiteral.elements().expressions().forEach(element -> checkAssignmentTarget(context, element));
      return;
    }
    Expression receiver = switch (expression) {
      case SubscriptionExpression subscription -> subscription.object();
      case QualifiedExpression qualified -> qualified.qualifier();
      default -> null;
    };
    if (receiver != null && FROZENDICT.isTrueFor(receiver, context)) {
      context.addIssue(receiver, MESSAGE);
    }
  }

  private static void checkCompoundAssignment(SubscriptionContext context) {
    checkAssignmentTarget(context, ((CompoundAssignmentStatement) context.syntaxNode()).lhsExpression());
  }

  private static void checkAnnotatedAssignment(SubscriptionContext context) {
    AnnotatedAssignment assignment = (AnnotatedAssignment) context.syntaxNode();
    if (assignment.assignedValue() != null) {
      checkAssignmentTarget(context, assignment.variable());
    }
  }
}
