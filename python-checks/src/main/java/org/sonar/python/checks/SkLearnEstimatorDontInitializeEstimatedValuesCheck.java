/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6974")
public class SkLearnEstimatorDontInitializeEstimatedValuesCheck extends PythonSubscriptionCheck {

  private static final String BASE_ESTIMATOR_FULLY_QUALIFIED_NAME = "sklearn.base.BaseEstimator";
  private static final Set<String> MIXINS_FULLY_QUALIFIED_NAME = Set.of(
    "sklearn.base.BiclusterMixin",
    "sklearn.base.ClassifierMixin",
    "sklearn.base.ClusterMixin",
    "sklearn.base.DensityMixin",
    "sklearn.base.MetaEstimatorMixin",
    "sklearn.base.OneToOneFeatureMixin",
    "sklearn.base.OutlierMixin",
    "sklearn.base.RegressorMixin",
    "sklearn.base.TransformerMixin");

  private static final String MESSAGE = "Move this estimated attribute in the `fit` method.";
  private static final String MESSAGE_SECONDARY = "The attribute is used in this estimator";
  public static final String QUICK_FIX_MESSAGE = "Remove the statement";
  public static final String QUICK_FIX_RENAME_MESSAGE = "Remove all trailing underscores from the variable name";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, SkLearnEstimatorDontInitializeEstimatedValuesCheck::checkFunction);
  }

  private static boolean inheritsMixin(ClassSymbol classSymbol) {
    return MIXINS_FULLY_QUALIFIED_NAME.stream().anyMatch(classSymbol::isOrExtends);
  }

  private static void checkFunction(SubscriptionContext subscriptionContext) {
    FunctionDefImpl functionDef = (FunctionDefImpl) subscriptionContext.syntaxNode();
    if (!"__init__".equals(functionDef.name().name())) {
      return;
    }

    var classDef = (ClassDef) TreeUtils.firstAncestorOfKind(functionDef, Tree.Kind.CLASSDEF);
    if (classDef == null) {
      return;
    }
    var classSymbol = TreeUtils.getClassSymbolFromDef(classDef);
    if (classSymbol == null) {
      return;
    }
    boolean inheritsBaseEstimator = Optional.of(classSymbol)
      .map(classSymbol1 -> classSymbol1.isOrExtends(BASE_ESTIMATOR_FULLY_QUALIFIED_NAME))
      .orElse(false);

    if (!inheritsBaseEstimator && !inheritsMixin(classSymbol)) {
      return;
    }

    var visitor = new VariableDeclarationEndingWithUnderscoreVisitor();
    functionDef.body().accept(visitor);
    var offendingVariables = visitor.qualifiedExpressions;
    var secondaryLocation = classDef.name();
    offendingVariables
      .forEach((qualifiedExpression, assignmentStatement) -> {
        var issue = subscriptionContext.addIssue(qualifiedExpression.name(), MESSAGE).secondary(secondaryLocation, MESSAGE_SECONDARY);

        createQuickFix(assignmentStatement).ifPresent(issue::addQuickFix);
        issue.addQuickFix(createQuickFixRename(qualifiedExpression));
      });
  }

  private static PythonQuickFix createQuickFixRename(QualifiedExpression qualifiedExpression) {
    var quickFix = PythonQuickFix.newQuickFix(QUICK_FIX_RENAME_MESSAGE);
    var newName = qualifiedExpression.name().name().replaceAll("_+$", "");
    return quickFix.addTextEdit(TextEditUtils.renameAllUsages(qualifiedExpression.name(), newName)).build();
  }

  private static Optional<PythonQuickFix> createQuickFix(AssignmentStatement assignmentStatement) {

    var builder = PythonQuickFix.newQuickFix(QUICK_FIX_MESSAGE);

    if (assignmentStatement.lhsExpressions().size() != 1 || assignmentStatement.lhsExpressions().stream().anyMatch(expressions -> expressions.expressions().size() != 1)) {
      return Optional.empty();
    }
    builder.addTextEdit(TextEditUtils.removeStatement(assignmentStatement));
    if (assignmentStatement.assignedValue().is(Tree.Kind.NONE)) {
      return Optional.of(builder.build());
    }
    return Optional.empty();
  }

  private static class VariableDeclarationEndingWithUnderscoreVisitor extends BaseTreeVisitor {

    private final Map<QualifiedExpression, AssignmentStatement> qualifiedExpressions = new HashMap<>();

    private static boolean isOffendingQualifiedExpression(QualifiedExpression qualifiedExpression) {
      return !qualifiedExpression.name().name().startsWith("__") && qualifiedExpression.name().name().endsWith("_") && qualifiedExpression.qualifier().is(Tree.Kind.NAME)
        && "self".equals(((org.sonar.plugins.python.api.tree.Name) qualifiedExpression.qualifier()).name());
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement pyAssignmentStatementTree) {
      var offendingQualifiedExpressions = pyAssignmentStatementTree.lhsExpressions()
        .stream()
        .flatMap(expressionList -> expressionList.expressions().stream())
        .filter(expression -> expression.is(Tree.Kind.QUALIFIED_EXPR))
        .map(QualifiedExpression.class::cast);

      var offendingTuples = pyAssignmentStatementTree.lhsExpressions()
        .stream()
        .flatMap(expressionList -> expressionList.expressions().stream())
        .filter(expression -> expression.is(Tree.Kind.TUPLE))
        .map(Tuple.class::cast)
        .flatMap(tuple -> tuple.elements().stream())
        .filter(expression -> expression.is(Tree.Kind.QUALIFIED_EXPR))
        .map(QualifiedExpression.class::cast);

      Stream.concat(
        offendingQualifiedExpressions, offendingTuples)
        .filter(VariableDeclarationEndingWithUnderscoreVisitor::isOffendingQualifiedExpression)
        .forEach(qualifiedExpression -> qualifiedExpressions.put(qualifiedExpression, pyAssignmentStatementTree));
      super.visitAssignmentStatement(pyAssignmentStatementTree);
    }
  }
}
