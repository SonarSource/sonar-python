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

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Symbol.Kind;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.CallExpressionImpl;
import org.sonar.python.tree.QualifiedExpressionImpl;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.CALL_EXPR;
import static org.sonar.python.tree.TreeUtils.toOptionalInstanceOfMapper;

@Rule(key = "S6973")
public class SklearnEstimatorHyperparametersCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Specify important hyperparameters when instantiating a Scikit-learn estimator.";

  private record Param(String name, Optional<Integer> position) {
    public Param(String name) {
      this(name, Optional.empty());
    }

    public Param(String name, int position) {
      this(name, Optional.of(position));
    }
  }

  private static final String LEARNING_RATE = "learning_rate";
  private static final String N_NEIGHBORS = "n_neighbors";
  private static final String KERNEL = "kernel";
  private static final String GAMMA = "gamma";
  private static final String C = "C";

  private static final Map<String, List<Param>> ESTIMATORS_AND_PARAMETERS_TO_CHECK = Map.ofEntries(
    Map.entry("sklearn.ensemble._weight_boosting.AdaBoostClassifier", List.of(new Param(LEARNING_RATE))),
    Map.entry("sklearn.ensemble._weight_boosting.AdaBoostRegressor", List.of(new Param(LEARNING_RATE))),
    Map.entry("sklearn.ensemble._gb.GradientBoostingClassifier", List.of(new Param(LEARNING_RATE))),
    Map.entry("sklearn.ensemble._gb.GradientBoostingRegressor", List.of(new Param(LEARNING_RATE))),
    Map.entry("sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier", List.of(new Param(LEARNING_RATE))),
    Map.entry("sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor", List.of(new Param(LEARNING_RATE))),
    Map.entry("sklearn.ensemble._forest.RandomForestClassifier", List.of(new Param("min_samples_leaf"), new Param("max_features"))),
    Map.entry("sklearn.ensemble._forest.RandomForestRegressor", List.of(new Param("min_samples_leaf"), new Param("max_features"))),
    Map.entry("sklearn.linear_model._coordinate_descent.ElasticNet", List.of(new Param("alpha", 0), new Param("l1_ratio"))),
    Map.entry("sklearn.neighbors._unsupervised.NearestNeighbors", List.of(new Param(N_NEIGHBORS, 0))),
    Map.entry("sklearn.neighbors._classification.KNeighborsClassifier", List.of(new Param(N_NEIGHBORS, 0))),
    Map.entry("sklearn.neighbors._regression.KNeighborsRegressor", List.of(new Param(N_NEIGHBORS, 0))),
    Map.entry("sklearn.svm._classes.NuSVC", List.of(new Param("nu"), new Param(KERNEL), new Param(GAMMA))),
    Map.entry("sklearn.svm._classes.NuSVR", List.of(new Param(C), new Param(KERNEL), new Param(GAMMA))),
    Map.entry("sklearn.svm._classes.SVC", List.of(new Param(C), new Param(KERNEL), new Param(GAMMA))),
    Map.entry("sklearn.svm._classes.SVR", List.of(new Param(C), new Param(KERNEL), new Param(GAMMA))),
    Map.entry("sklearn.tree._classes.DecisionTreeClassifier", List.of(new Param("ccp_alpha"))),
    Map.entry("sklearn.tree._classes.DecisionTreeRegressor", List.of(new Param("ccp_alpha"))),
    Map.entry("sklearn.neural_network._multilayer_perceptron.MLPClassifier", List.of(new Param("hidden_layer_sizes", 0))),
    Map.entry("sklearn.neural_network._multilayer_perceptron.MLPRegressor", List.of(new Param("hidden_layer_sizes", 0))),
    Map.entry("sklearn.preprocessing._polynomial.PolynomialFeatures", List.of(new Param("degree", 0), new Param("interaction_only"))));

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(CALL_EXPR, SklearnEstimatorHyperparametersCheck::checkEstimator);
  }

  private static void checkEstimator(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();

    Symbol calleeSymbol = callExpression.calleeSymbol();

    Optional.ofNullable(calleeSymbol)
      .filter(callee -> callee.is(Kind.CLASS))
      .map(ClassSymbol.class::cast)
      .map(ClassSymbol::fullyQualifiedName)
      .map(ESTIMATORS_AND_PARAMETERS_TO_CHECK::get)
      .filter(parameters -> !isSetParamsCalled(callExpression))
      .filter(parameters -> !isPartOfPipelineAndGridSearch(callExpression))
      .filter(parameters -> isMissingAHyperparameter(callExpression, parameters))
      .ifPresent(parameters -> ctx.addIssue(callExpression, MESSAGE));
  }

  private static boolean isMissingAHyperparameter(CallExpression callExpression, List<Param> parametersToCheck) {
    return parametersToCheck.stream()
      .map(param -> {
        if (param.position.isPresent()) {
          return TreeUtils.nthArgumentOrKeyword(param.position.get(), param.name, callExpression.arguments());
        }
        return TreeUtils.argumentByKeyword(param.name, callExpression.arguments());
      })
      .anyMatch(Objects::isNull);
  }

  private static boolean isPartOfPipelineAndGridSearch(CallExpression callExpression) {
    return getPipelineAssignement(callExpression)
      .map(SklearnEstimatorHyperparametersCheck::isPipelineUsedInGridSearch)
      .orElse(false);
  }

  private static Optional<Name> getPipelineAssignement(CallExpression callExpression) {
    return Optional.ofNullable(TreeUtils.firstAncestorOfKind(callExpression, CALL_EXPR))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpressionImpl.class))
      .filter(Objects::nonNull)
      .filter(callExp -> Optional.ofNullable(callExp.calleeSymbol())
        .map(Symbol::fullyQualifiedName)
        .map(fqn -> "sklearn.pipeline.make_pipeline".equals(fqn) || "sklearn.pipeline.Pipeline".equals(fqn))
        .orElse(false))
      .flatMap(Expressions::getAssignedName);
  }

  private static boolean isPipelineUsedInGridSearch(Name pipeline) {
    return Optional.ofNullable(pipeline.symbol())
      .map(Symbol::usages)
      .map(usages -> usages.stream()
        .map(Usage::tree)
        .map(Tree::parent)
        .filter(parent -> parent.is(Tree.Kind.REGULAR_ARGUMENT))
        .map(RegularArgument.class::cast)
        .anyMatch(SklearnEstimatorHyperparametersCheck::isArgumentPartOfGridSearch))
      .orElse(false);
  }

  private static boolean isArgumentPartOfGridSearch(RegularArgument arg) {
    return Optional.ofNullable(TreeUtils.firstAncestorOfKind(arg, CALL_EXPR))
      .flatMap(toOptionalInstanceOfMapper(CallExpressionImpl.class))
      .map(CallExpressionImpl::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .map("sklearn.model_selection._search.GridSearchCV"::equals)
      .orElse(false);
  }

  private static boolean isSetParamsCalled(CallExpression callExpression) {
    return Expressions.getAssignedName(callExpression)
      .map(Name::symbol)
      .map(Symbol::usages)
      .map(SklearnEstimatorHyperparametersCheck::isUsedWithSetParams)
      .orElse(false);
  }

  private static boolean isUsedWithSetParams(List<Usage> usages) {
    return usages.stream()
      .map(Usage::tree)
      .map(Tree::parent)
      .filter(parent -> parent.is(Tree.Kind.QUALIFIED_EXPR))
      .map(TreeUtils.toInstanceOfMapper(QualifiedExpressionImpl.class))
      .filter(Objects::nonNull)
      .map(qExp -> qExp.name().name())
      .anyMatch("set_params"::equals);
  }

}
