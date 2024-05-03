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

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.DictionaryLiteralImpl;
import org.sonar.python.tree.KeyValuePairImpl;
import org.sonar.python.tree.ListLiteralImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.tree.TupleImpl;

@Rule(key = "S6972")
public class SklearnPipelineParameterAreCorrectCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Provide valid parameters to the estimator.";
  private static final Set<String> SKLEARN_SEARCH_FQNS = Set.of(
    "sklearn.model_selection._search.GridSearchCV",
    "sklearn.model_selection._search.HalvingGridSearchCV",
    "sklearn.model_selection._search.RandomizedSearchCV",
    "sklearn.model_selection._search.HalvingRandomSearchCV");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, SklearnPipelineParameterAreCorrectCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();

    var parsedFunctionOptional = Optional.ofNullable(callExpression.calleeSymbol())
      .map(Symbol::fullyQualifiedName)
      .filter(SKLEARN_SEARCH_FQNS::contains)
      .map(callExpr -> getStepAndParametersFromDict(callExpression))
      .flatMap(parsedParameters -> getPipelineNameAndParsedParametersFromSearchFunctions(parsedParameters, callExpression))
      .or(() -> Optional.ofNullable(callExpression.calleeSymbol())
        .map(Symbol::fullyQualifiedName)
        .filter("sklearn.pipeline.Pipeline.set_params"::equals)
        .map(callExpr -> getStepAndParametersFromArguments(callExpression))
        .flatMap(parsedParameters -> getPipelineNameAndParsedParametersFromPipelineSetParamsFunction(parsedParameters, callExpression)));

    parsedFunctionOptional.ifPresent(
      pipelineNameAndParsedParameters -> {
        var parsedFunction = pipelineNameAndParsedParameters.parsedParameters();
        var pipelineName = pipelineNameAndParsedParameters.pipelineName();
        Expressions.singleAssignedNonNameValue(pipelineName)
          .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
          .ifPresent(pipelineCallExpr -> findProblems(parsedFunction, parsePipeline(pipelineCallExpr), subscriptionContext));
      });
  }

  private static Optional<PipelineNameAndParsedParameters> getPipelineNameAndParsedParametersFromPipelineSetParamsFunction(Map<String, Set<StringAndTree>> parsedParameters,
    CallExpression callExpression) {
    return newPipelineNameAndParsedParameters(
      Optional.of(callExpression).map(CallExpression::callee).flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
        .map(QualifiedExpression::qualifier).flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class)).orElse(null),
      parsedParameters);
  }

  private static Optional<PipelineNameAndParsedParameters> getPipelineNameAndParsedParametersFromSearchFunctions(Map<String, Set<StringAndTree>> parsedParameters,
    CallExpression callExpression) {
    return newPipelineNameAndParsedParameters(
      Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, "estimator", callExpression.arguments()))
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
        .map(RegularArgument::expression)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
        .orElse(null),
      parsedParameters);
  }

  private record PipelineNameAndParsedParameters(Name pipelineName, Map<String, Set<StringAndTree>> parsedParameters) {
  }

  private static Optional<PipelineNameAndParsedParameters> newPipelineNameAndParsedParameters(@Nullable Name pipelineName, Map<String, Set<StringAndTree>> parsedParameters) {
    return Optional.ofNullable(pipelineName).map(pipelineName1 -> new PipelineNameAndParsedParameters(pipelineName1, parsedParameters));
  }

  private static void findProblems(Map<String, Set<StringAndTree>> setParameters, Map<String, ClassSymbol> pipelineDefinition, SubscriptionContext subscriptionContext) {
    for (var entry : setParameters.entrySet()) {
      var step = entry.getKey();
      var stringAndTree = entry.getValue();
      var parameters = stringAndTree.stream().map(StringAndTree::string).collect(Collectors.toSet());

      var classifier = pipelineDefinition.get(step);
      if (classifier == null) {
        // Maybe quickfix ?
        continue;
      }
      var possibleParameters = getInitFunctionSymbol(classifier).map(FunctionSymbol::parameters).orElse(List.of());

      parameters.forEach(parameter -> {
        if (isNotAValidParameter(parameter, possibleParameters)) {
          createIssue(subscriptionContext, parameter, stringAndTree);
        }
      });
    }
  }

  private static void createIssue(SubscriptionContext subscriptionContext, String parameter, Set<StringAndTree> stringAndTree) {
    stringAndTree.stream().filter(stringAndTree1 -> stringAndTree1.string().equals(parameter)).findFirst()
      .ifPresent(location -> subscriptionContext.addIssue(location.tree, MESSAGE));
  }

  private static boolean isNotAValidParameter(String parameter, List<FunctionSymbol.Parameter> possibleParameters) {
    return possibleParameters.stream().noneMatch(symbol -> Objects.equals(symbol.name(), parameter));
  }

  private static Optional<FunctionSymbol> getInitFunctionSymbol(ClassSymbol classSymbol) {
    return classSymbol.declaredMembers().stream().filter(
      memberSymbol -> "__init__".equals(memberSymbol.name())).findFirst().map(FunctionSymbol.class::cast);
  }

  private static Stream<Expression> getExpressionsFromArgument(@Nullable RegularArgument argument) {
    return Optional.ofNullable(argument)
      .map(RegularArgument::expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(ListLiteralImpl.class))
      .map(ListLiteralImpl::elements)
      .map(ExpressionList::expressions)
      .stream()
      .flatMap(Collection::stream);
  }

  private static Map<String, ClassSymbol> parsePipeline(CallExpression callExpression) {
    var stepsArgument = TreeUtils.nthArgumentOrKeyword(0, "steps", callExpression.arguments());
    var out = new HashMap<String, ClassSymbol>();

    getExpressionsFromArgument(stepsArgument)
      .map(
        TreeUtils.toInstanceOfMapper(TupleImpl.class))
      .filter(Objects::nonNull)
      .map(TupleImpl::elements)
      .filter(elements -> elements.size() == 2)
      .forEach(
        tuple -> getResult(tuple).ifPresent(result1 -> out.put(result1.stepName(), result1.classifierName())));
    return out;
  }

  private static Optional<Result> getResult(List<Expression> tuple) {
    var step = tuple.get(0);
    var classifier = tuple.get(1);
    var stepName = Optional.ofNullable(step)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .map(StringLiteral::trimmedQuotesValue);
    var classifierName = Optional.ofNullable(classifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .map(CallExpression::calleeSymbol)
      .map(ClassSymbol.class::cast);

    return stepName.flatMap(stepName1 -> classifierName.map(classifierName1 -> new Result(stepName1, classifierName1)));
  }

  private record Result(String stepName, ClassSymbol classifierName) {
  }

  private static Map<String, Set<StringAndTree>> getStepAndParametersFromArguments(CallExpression callExpression) {
    return callExpression.arguments().stream().filter(argument -> argument.is(Tree.Kind.REGULAR_ARGUMENT))
      .map(RegularArgument.class::cast)
      .map(RegularArgument::keywordArgument)
      .filter(Objects::nonNull)
      .map(SklearnPipelineParameterAreCorrectCheck::getStepAndParameterFromName)
      .<StepAndParameter>mapMulti(Optional::ifPresent)
      .collect(mergeStringAndTreeToMapCollector());
  }

  private static Map<String, Set<StringAndTree>> getStepAndParametersFromDict(CallExpression callExpression) {
    return Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(1, "param_grid", callExpression.arguments()))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
      .map(RegularArgument::expression)
      .stream()
      .flatMap(SklearnPipelineParameterAreCorrectCheck::extractKeyValuePairFromDictLiteral)
      .map(KeyValuePairImpl::key)
      .map(TreeUtils.toInstanceOfMapper(StringLiteral.class))
      .filter(Objects::nonNull)
      .map(stringLiteral -> getStepAndParameterFromString(stringLiteral.trimmedQuotesValue(), stringLiteral))
      .<StepAndParameter>mapMulti(Optional::ifPresent)
      .collect(
        mergeStringAndTreeToMapCollector());
  }

  private static Stream<KeyValuePairImpl> extractKeyValuePairFromDictLiteral(Expression expression) {
    return Optional.of(expression)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .flatMap(Expressions::singleAssignedNonNameValue)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(DictionaryLiteralImpl.class))
      .map(DictionaryLiteral::elements)
      .stream()
      .flatMap(Collection::stream)
      .map(TreeUtils.toInstanceOfMapper(KeyValuePairImpl.class))
      .filter(Objects::nonNull);
  }

  private static Collector<StepAndParameter, ?, Map<String, Set<StringAndTree>>> mergeStringAndTreeToMapCollector() {
    return Collectors.toMap(StepAndParameter::step, stepAndParameter -> Set.of(new StringAndTree(stepAndParameter.parameter, stepAndParameter.location)),
      (set1, set2) -> {
        var set = new HashSet<>(set1);
        set.addAll(set2);
        return set;
      });
  }

  private record StringAndTree(String string, Tree tree) {
  }

  private static Optional<StepAndParameter> getStepAndParameterFromName(Name name) {
    var split = name.name().split("__");
    if (split.length != 2) {
      return Optional.empty();
    }
    return Optional.of(new StepAndParameter(split[0], split[1], name));
  }

  private static Optional<StepAndParameter> getStepAndParameterFromString(String string, Tree location) {
    var split = string.split("__");
    if (split.length != 2 || string.endsWith("__")) {
      return Optional.empty();
    }
    return Optional.of(new StepAndParameter(split[0], split[1], location));
  }

  private record StepAndParameter(String step, String parameter, Tree location) {
  }
}
