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

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.tree.Tree.Kind.CALL_EXPR;
import static org.sonar.plugins.python.api.tree.Tree.Kind.REGULAR_ARGUMENT;
import static org.sonar.python.tree.TreeUtils.toOptionalInstanceOfMapper;

@Rule(key = "S6973")
public class MissingHyperParameterCheck extends PythonSubscriptionCheck {
  private static final String SKLEARN_MESSAGE = "Add the missing hyperparameter%s %s for this Scikit-learn estimator.";
  private static final String PYTORCH_MESSAGE = "Add the missing hyperparameter%s %s for this PyTorch optimizer.";

  private record Param(String name, Optional<Integer> position) {
    public Param(String name) {
      this(name, Optional.empty());
    }

    public Param(String name, int position) {
      this(name, Optional.of(position));
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(CALL_EXPR, MissingHyperParameterCheck::checkEstimator);
  }

  private static void checkEstimator(SubscriptionContext ctx) {
    CallExpression callExpression = (CallExpression) ctx.syntaxNode();
    Symbol calleeSymbol = callExpression.calleeSymbol();

    Optional.ofNullable(calleeSymbol)
      .map(Symbol::fullyQualifiedName).ifPresent(name -> {
        checkPyTorchOptimizer(name, callExpression, ctx);
        checkSkLearnEstimator(name, callExpression, ctx);
      });
  }

  private static void checkPyTorchOptimizer(String name, CallExpression callExpression, SubscriptionContext ctx) {
    List<String> missingParams = PyTorchCheck.getMissingParameters(name, callExpression).stream()
      .map(Param::name)
      .toList();

    if (!missingParams.isEmpty()) {
      ctx.addIssue(callExpression, formatMessage(missingParams, PYTORCH_MESSAGE));
    }
  }

  private static void checkSkLearnEstimator(String name, CallExpression callExpression, SubscriptionContext ctx) {
    List<String> missingParams = SkLearnCheck.getMissingParameters(name, callExpression).stream().map(Param::name).toList();
    if(!missingParams.isEmpty()) {
      ctx.addIssue(callExpression, formatMessage(missingParams, SKLEARN_MESSAGE));
    }
  }

  private static String formatMessage(List<String> missingArgs, String formatString) {
    String plural = missingArgs.size() == 1 ? "" : "s";
    String missingArgsString = missingArgs.get(missingArgs.size() - 1);
    if (missingArgs.size() > 1) {
      missingArgsString = missingArgs.subList(0, missingArgs.size() - 1).stream().collect(Collectors.joining(", " +
        "")) + " and " + missingArgsString;
    }
    return formatString.formatted(plural, missingArgsString);
  }


  // common method used by both the PyTorchCheck class and SkLearnCheck class
  private static List<Param> filterUsedHyperparameter(CallExpression callExpression, List<Param> parametersToCheck) {
    return parametersToCheck.stream()
      .filter(param -> param.position()
        .map(position -> TreeUtils.nthArgumentOrKeyword(position, param.name, callExpression.arguments()))
        .orElse(TreeUtils.argumentByKeyword(param.name, callExpression.arguments())) == null)
      .toList();
  }

  private static class PyTorchCheck {
    public static final String LR = "lr";
    public static final String WEIGHT_DECAY = "weight_decay";

    private static final Map<String, List<Param>> PY_TORCH_ESTIMATORS_AND_PARAMETERS_TO_CHECK = Map.ofEntries(
      Map.entry("torch.utils.data.DataLoader", List.of(new Param("batch_size", 1))),
      Map.entry("torch.optim.Adadelta", List.of(new Param(LR, 1), new Param(WEIGHT_DECAY, 4))),
      Map.entry("torch.optim.Adagrad", List.of(new Param(LR, 1), new Param(WEIGHT_DECAY, 3))),
      Map.entry("torch.optim.Adam", List.of(new Param(LR, 1), new Param(WEIGHT_DECAY, 4))),
      Map.entry("torch.optim.AdamW", List.of(new Param(LR, 1), new Param(WEIGHT_DECAY, 4))),
      Map.entry("torch.optim.SparseAdam", List.of(new Param(LR, 1))),
      Map.entry("torch.optim.Adamax", List.of(new Param(LR, 1), new Param(WEIGHT_DECAY, 4))),
      Map.entry("torch.optim.ASGD", List.of(new Param(LR, 1), new Param(WEIGHT_DECAY, 5))),
      Map.entry("torch.optim.LBFGS", List.of(new Param(LR, 1))),
      Map.entry("torch.optim.NAdam", List.of(new Param(LR, 1), new Param(WEIGHT_DECAY, 4), new Param("momentum_decay", 5))),
      Map.entry("torch.optim.RAdam", List.of(new Param(LR, 1), new Param(WEIGHT_DECAY, 4))),
      Map.entry("torch.optim.RMSprop", List.of(new Param(LR, 1), new Param(WEIGHT_DECAY, 4), new Param("momentum", 5))),
      Map.entry("torch.optim.Rprop", List.of(new Param(LR, 1))),
      Map.entry("torch.optim.SGD", List.of(new Param(LR, 1), new Param("momentum", 2), new Param(WEIGHT_DECAY, 4)))
    );

    public static List<Param> getMissingParameters(String name, CallExpression callExpression) {
      return Optional.ofNullable(PY_TORCH_ESTIMATORS_AND_PARAMETERS_TO_CHECK.get(name))
        .filter(parameters -> !Expressions.containsSpreadOperator(callExpression.arguments()))
        .map(parameters -> filterUsedHyperparameter(callExpression, parameters))
        .orElse(Collections.emptyList());
    }
  }

  private static class SkLearnCheck {
    private static final String LEARNING_RATE = "learning_rate";
    private static final String N_NEIGHBORS = "n_neighbors";
    private static final String KERNEL = "kernel";
    private static final String GAMMA = "gamma";
    private static final String C = "C";

    private static final Map<String, List<Param>> SK_LEARN_ESTIMATORS_AND_PARAMETERS_TO_CHECK = Map.ofEntries(
      Map.entry("sklearn.ensemble._weight_boosting.AdaBoostClassifier", List.of(new Param(LEARNING_RATE))),
      Map.entry("sklearn.ensemble._weight_boosting.AdaBoostRegressor", List.of(new Param(LEARNING_RATE))),
      Map.entry("sklearn.ensemble._gb.GradientBoostingClassifier", List.of(new Param(LEARNING_RATE))),
      Map.entry("sklearn.ensemble._gb.GradientBoostingRegressor", List.of(new Param(LEARNING_RATE))),
      Map.entry("sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier",
        List.of(new Param(LEARNING_RATE))),
      Map.entry("sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor",
        List.of(new Param(LEARNING_RATE))),
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

    private static final Set<String> SEARCH_CV_FQNS = Set.of(
      "sklearn.model_selection._search.GridSearchCV",
      "sklearn.model_selection._search.RandomizedSearchCV",
      "sklearn.model_selection._search_successive_halving.HalvingRandomSearchCV",
      "sklearn.model_selection._search_successive_halving.HalvingGridSearchCV");

    public static List<Param> getMissingParameters(String name, CallExpression callExpression) {
      return Optional.ofNullable(SK_LEARN_ESTIMATORS_AND_PARAMETERS_TO_CHECK.get(name))
        .filter(parameters -> !isDirectlyUsedInSearchCV(callExpression))
        .filter(parameters -> !isSetParamsCalled(callExpression))
        .filter(parameters -> !isPartOfPipelineAndSearchCV(callExpression))
        .map(parameters -> filterUsedHyperparameter(callExpression, parameters))
        .orElse(Collections.emptyList());
    }

    private static boolean isDirectlyUsedInSearchCV(CallExpression callExpression) {
      Tree current = callExpression;
      do{
        current = TreeUtils.firstAncestorOfKind(current, REGULAR_ARGUMENT);
        if(current instanceof RegularArgument arg && isArgumentPartOfSearchCV(arg)) {
          return true;
        }
      } while(current != null);
      return false;
    }

    private static boolean isSetParamsCalled(CallExpression callExpression) {
      return Expressions.getAssignedName(callExpression)
        .map(Name::symbol)
        .map(Symbol::usages)
        .map(SkLearnCheck::isUsedWithSetParams)
        .orElse(false);
    }

    private static boolean isUsedWithSetParams(List<Usage> usages) {
      return usages.stream()
        .map(Usage::tree)
        .map(Tree::parent)
        .filter(parent -> parent.is(Tree.Kind.QUALIFIED_EXPR))
        .map(TreeUtils.toInstanceOfMapper(QualifiedExpression.class))
        .filter(Objects::nonNull)
        .map(qExp -> qExp.name().name())
        .anyMatch("set_params"::equals);
    }

    private static boolean isPartOfPipelineAndSearchCV(CallExpression callExpression) {
      return Expressions.getAssignedName(callExpression)
        .map(SkLearnCheck::isEstimatorUsedInSearchCV)
        .orElse(false);
    }

    private static boolean isEstimatorUsedInSearchCV(Name estimator) {
      return Optional.ofNullable(estimator.symbol())
        .map(Symbol::usages)
        .map(usages -> usages.stream()
          .map(Usage::tree)
          .map(Tree::parent)
          .filter(parent -> parent.is(REGULAR_ARGUMENT))
          .map(RegularArgument.class::cast)
          .anyMatch(SkLearnCheck::isArgumentPartOfSearchCV))
        .orElse(false);
    }

    private static boolean isArgumentPartOfSearchCV(RegularArgument arg) {
      return Optional.ofNullable(TreeUtils.firstAncestorOfKind(arg, CALL_EXPR))
        .flatMap(toOptionalInstanceOfMapper(CallExpression.class))
        .map(CallExpression::calleeSymbol)
        .map(Symbol::fullyQualifiedName)
        .map(SEARCH_CV_FQNS::contains)
        .orElse(false);
    }
  }
}
