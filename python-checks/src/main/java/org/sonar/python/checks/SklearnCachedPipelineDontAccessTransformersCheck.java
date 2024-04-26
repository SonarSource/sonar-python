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
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.Stream;
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
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.tree.TupleImpl;

@Rule(key = "S6971")
public class SklearnCachedPipelineDontAccessTransformersCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Avoid accessing transformers in a cached pipeline.";
  public static final String MESSAGE_SECONDARY = "Accessed here";
  public static final String MESSAGE_SECONDARY_CREATION = "Pipeline created here";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, SklearnCachedPipelineDontAccessTransformersCheck::checkCallExpr);
  }

  private static void checkCallExpr(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();
    PipelineCreation pipelineCreation = isPipelineCreation(callExpression);
    if (pipelineCreation == PipelineCreation.NOT_PIPELINE) {
      return;
    }
    var memoryArgument = TreeUtils.argumentByKeyword("memory", callExpression.arguments());
    if (memoryArgument == null || memoryArgument.expression().is(Tree.Kind.NONE)) {
      return;
    }
    var stepsArgument = TreeUtils.nthArgumentOrKeyword(0, "steps", callExpression.arguments());

    Stream<Name> nameStream;
    Map<Name, String> nameToStepName = new HashMap<>();
    Optional<Expression> stepArgumentExpression = Optional.ofNullable(stepsArgument)
      .map(RegularArgument::expression);

    if (pipelineCreation == PipelineCreation.PIPELINE) {
      var tuples = stepArgumentExpression
        .filter(e -> e.is(Tree.Kind.LIST_LITERAL))
        .map(e -> ((ListLiteral) e).elements().expressions())
        .stream()
        .flatMap(Collection::stream)
        .filter(e -> e.is(Tree.Kind.TUPLE));

      nameStream = tuples
        .map(t -> ((TupleImpl) t).elements())
        .filter(e -> e.size() == 2)
        .filter(e -> e.get(1).is(Tree.Kind.NAME))
        .map(e2 -> {
          if (e2.get(0).is(Tree.Kind.STRING_LITERAL)) {
            nameToStepName.put((Name) e2.get(1), ((StringLiteral) e2.get(0)).trimmedQuotesValue());
          }
          return e2;
        })
        .map(e -> e.get(1))
        .map(Name.class::cast);
    } else {
      nameStream = stepArgumentExpression
        .filter(e -> e.is(Tree.Kind.NAME))
        .map(Name.class::cast)
        .stream();
    }
    var qualifiedUses = nameStream
      .collect(Collectors.toMap(n -> n, SklearnCachedPipelineDontAccessTransformersCheck::symbolIsUsedInQualifiedExpression));

    qualifiedUses.forEach((name, uses) -> {
      if (!uses.isEmpty()) {
        var issue = subscriptionContext.addIssue(name, MESSAGE);
        uses.forEach((useTree, qualExpr) -> issue.secondary(useTree, MESSAGE_SECONDARY));
        if (pipelineCreation == PipelineCreation.PIPELINE) {
          issue.secondary(callExpression.callee(), MESSAGE_SECONDARY_CREATION);
          uses
            .forEach((useTree, qualExpr) -> getAssignedName(callExpression).flatMap(pipelineBindingVariable -> getQuickFix(pipelineBindingVariable, name, qualExpr, nameToStepName))
              .ifPresent(issue::addQuickFix));
        }
      }
    });
  }

  private static Optional<PythonQuickFix> getQuickFix(Name pipelineBindingVariable, Tree name, QualifiedExpression qualifiedExpression, Map<Name, String> nameToStepName) {
    var quickFix = PythonQuickFix.newQuickFix("Access the property through the ");
    String stepName = nameToStepName.get(name);
    if (stepName == null) {
      return Optional.empty();
    }
    quickFix.addTextEdit(TextEditUtils.replace(qualifiedExpression.qualifier(), String.format("%s.named_steps[\"%s\"]", pipelineBindingVariable.name(), stepName)));
    return Optional.of(quickFix.build());
  }

  private static Map<Tree, QualifiedExpression> symbolIsUsedInQualifiedExpression(Name name) {
    Symbol symbol = name.symbol();
    if (symbol == null) {
      return new HashMap<>();
    }
    Map<Tree, QualifiedExpression> qualifiedExpressions = new HashMap<>();
    symbol.usages().stream()
      .filter(u -> u.tree().parent().is(Tree.Kind.QUALIFIED_EXPR))
      .forEach(u -> qualifiedExpressions.put(((QualifiedExpression) u.tree().parent()).qualifier(), ((QualifiedExpression) u.tree().parent())));

    return qualifiedExpressions;
  }

  private enum PipelineCreation {
    PIPELINE,
    MAKE_PIPELINE,
    NOT_PIPELINE
  }

  private static PipelineCreation isPipelineCreation(CallExpression callExpression) {
    return Optional.ofNullable(callExpression.calleeSymbol()).map(Symbol::fullyQualifiedName)
      .map(fqn -> {
        if ("sklearn.pipeline.Pipeline".equals(fqn)) {
          return PipelineCreation.PIPELINE;
        }
        if ("sklearn.pipeline.make_pipeline".equals(fqn)) {
          return PipelineCreation.MAKE_PIPELINE;
        }
        return PipelineCreation.NOT_PIPELINE;
      }).orElse(PipelineCreation.NOT_PIPELINE);
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
}
