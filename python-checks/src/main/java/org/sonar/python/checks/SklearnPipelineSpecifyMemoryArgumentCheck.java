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

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6969")
public class SklearnPipelineSpecifyMemoryArgumentCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Specify a memory argument for the pipeline.";
  public static final String MESSAGE_QUICKFIX = "Add the memory argument";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, SklearnPipelineSpecifyMemoryArgumentCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext subscriptionContext) {
    Optional.of(subscriptionContext.syntaxNode())
      .map(CallExpression.class::cast)
      .filter(SklearnPipelineSpecifyMemoryArgumentCheck::isPipelineCreation)
      .ifPresent(
        callExpression -> {
          var memoryArgument = TreeUtils.argumentByKeyword("memory", callExpression.arguments());

          if (memoryArgument != null) {
            return;
          }

          if (getAssignedName(callExpression).map(SklearnPipelineSpecifyMemoryArgumentCheck::isUsedInAnotherPipeline).orElse(false)) {
            return;
          }

          createIssue(subscriptionContext, callExpression);
        });
  }

  private static void createIssue(SubscriptionContext subscriptionContext, CallExpression callExpression) {
    var issue = subscriptionContext.addIssue(callExpression.callee(), MESSAGE);
    var quickFix = PythonQuickFix.newQuickFix(MESSAGE_QUICKFIX)
      .addTextEdit(TextEditUtils.insertBefore(callExpression.rightPar(), ", memory=None"))
      .build();
    issue.addQuickFix(quickFix);
  }

  private static Optional<Name> getAssignedName(Expression expression) {
    if (expression.is(Tree.Kind.NAME)) {
      return Optional.of((Name) expression);
    }
    if (expression.is(Tree.Kind.QUALIFIED_EXPR)) {
      return getAssignedName(((QualifiedExpression) expression).name());
    }
    var assignment = (AssignmentStatement) TreeUtils.firstAncestorOfKind(expression, Tree.Kind.ASSIGNMENT_STMT);
    if (assignment == null) {
      return Optional.empty();
    }
    var expressions = SymbolUtils.assignmentsLhs(assignment);
    if (expressions.size() != 1) {
      List<Expression> rhsExpressions = getExpressionsFromRhs(assignment.assignedValue());
      var rhsIndex = rhsExpressions.indexOf(expression);
      if (rhsIndex != -1) {
        return getAssignedName(expressions.get(rhsIndex));
      } else {
        return Optional.empty();
      }
    }
    return getAssignedName(expressions.get(0));
  }

  private static List<Expression> getExpressionsFromRhs(Expression rhs) {
    List<Expression> expressions = new ArrayList<>();
    if (rhs.is(Tree.Kind.TUPLE)) {
      expressions.addAll(((Tuple) rhs).elements());
    } else if (rhs.is(Tree.Kind.LIST_LITERAL)) {
      expressions.addAll(((ListLiteral) rhs).elements().expressions());
    } else if (rhs.is(Tree.Kind.UNPACKING_EXPR)) {
      return getExpressionsFromRhs(((UnpackingExpression) rhs).expression());
    }
    return expressions;
  }

  private static boolean isPipelineCreation(CallExpression callExpression) {
    return Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .map(fqn -> "sklearn.pipeline.Pipeline".equals(fqn) || "sklearn.pipeline.make_pipeline".equals(fqn))
      .orElse(false);
  }

  private static boolean isUsedInAnotherPipeline(Name name) {
    Symbol symbol = name.symbol();
    return symbol != null && symbol.usages().stream().filter(usage -> !usage.isBindingUsage()).anyMatch(u -> {
      Tree tree = u.tree();
      CallExpression callExpression = (CallExpression) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.CALL_EXPR);
      while (callExpression != null) {
        Optional<String> fullyQualifiedName = Optional.ofNullable(callExpression.calleeSymbol()).map(Symbol::fullyQualifiedName);
        if (fullyQualifiedName.isPresent() && isPipelineCreation(callExpression)) {
          return true;
        }
        callExpression = (CallExpression) TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.CALL_EXPR);
      }
      return false;
    });
  }
}
