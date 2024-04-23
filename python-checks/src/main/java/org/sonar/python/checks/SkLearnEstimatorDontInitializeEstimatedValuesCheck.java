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

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6794")
public class SkLearnEstimatorDontInitializeEstimatedValuesCheck extends PythonSubscriptionCheck {

  private static final Set<String> BASE_FULLY_QUALIFIED_NAMES = Set.of(
    "sklearn.base.BaseEstimator",
    "sklearn.base.ClassifierMixin",
    "sklearn.base.RegressorMixin",
    "sklearn.base.TransformerMixin",
    "sklearn.base.ClusterMixin",
    "sklearn.base.DensityMixin",
    "sklearn.base.MetaEstimatorMixin",
    "sklearn.base.BiclusterMixin",
    "sklearn.base.OneToOneFeatureMixin",
    "sklearn.base.OutlierMixin");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, SkLearnEstimatorDontInitializeEstimatedValuesCheck::checkFunction);
  }

  private static void checkFunction(SubscriptionContext subscriptionContext) {
    FunctionDefImpl functionDef = (FunctionDefImpl) subscriptionContext.syntaxNode();
    if (!"__init__".equals(functionDef.name().name())) {
      return;
    }

    boolean inheritsBaseEstimator = Optional.ofNullable(TreeUtils.firstAncestorOfKind(functionDef, Tree.Kind.CLASSDEF))
      .map(ClassDef.class::cast)
      .map(TreeUtils::getClassSymbolFromDef)
      .map(ClassSymbol::superClasses)
      .map(superClasses -> superClasses.stream().anyMatch(symbol -> BASE_FULLY_QUALIFIED_NAMES.contains(symbol.fullyQualifiedName())))
      .orElse(false);

    if (!inheritsBaseEstimator) {
      return;
    }

    var visitor = new VariableDeclarationEndingWithUnderscoreVisitor();
    functionDef.body().accept(visitor);
    var offendingVariables = visitor.qualifiedExpressions;
    offendingVariables
      .forEach(qualifiedExpression -> subscriptionContext.addIssue(qualifiedExpression.name(), "Remove this initialization of an estimated value ending with an underscore."));

  }

  private static class VariableDeclarationEndingWithUnderscoreVisitor extends BaseTreeVisitor {

    private final Set<QualifiedExpression> qualifiedExpressions = new HashSet<>();

    @Override
    public void visitAssignmentStatement(AssignmentStatement pyAssignmentStatementTree) {
      var a = pyAssignmentStatementTree.lhsExpressions()
        .stream()
        .flatMap(expressionList -> expressionList.expressions().stream())
        .filter(expression -> expression.is(Tree.Kind.QUALIFIED_EXPR))
        .map(QualifiedExpression.class::cast)
        .filter(qualifiedExpression -> qualifiedExpression.name().name().endsWith("_") && qualifiedExpression.qualifier().is(Tree.Kind.NAME)
          && "self".equals(((org.sonar.plugins.python.api.tree.Name) qualifiedExpression.qualifier()).name()))
        .toList();

      this.qualifiedExpressions.addAll(a);
      super.visitAssignmentStatement(pyAssignmentStatementTree);
    }
  }
}
