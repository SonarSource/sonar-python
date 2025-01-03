/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
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
    "sklearn.model_selection._search_successive_halving.HalvingGridSearchCV",
    "sklearn.model_selection._search.RandomizedSearchCV",
    "sklearn.model_selection._search_successive_halving.HalvingRandomSearchCV");

  private record PipelineNameAndParsedParameters(Name pipelineName, Map<String, Set<ParameterNameAndLocation>> parsedParameters) {
  }
  private record ExpressionAndPrefix(List<Expression> tuple, String prefix, int depth) {
  }
  private record ParameterNameAndLocation(String string, Tree tree) {
  }
  private record StepAndClassifier(String stepName, ClassSymbol classifierName) {
  }
  private record StepAndParameter(String step, String parameter, Tree location) {
  }

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
        var parsedFunction = pipelineNameAndParsedParameters.parsedParameters;
        var pipelineName = pipelineNameAndParsedParameters.pipelineName;
        Expressions.singleAssignedNonNameValue(pipelineName)
          .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
          .ifPresent(pipelineCallExpr -> findProblems(parsedFunction, parsePipeline(pipelineCallExpr), subscriptionContext));
      });
  }

  private static Optional<PipelineNameAndParsedParameters> getPipelineNameAndParsedParametersFromPipelineSetParamsFunction(
    Map<String, Set<ParameterNameAndLocation>> parsedParameters,
    CallExpression callExpression) {
    return newPipelineNameAndParsedParameters(
      Optional.of(callExpression).map(CallExpression::callee)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
        .map(QualifiedExpression::qualifier)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
        .orElse(null),
      parsedParameters);
  }

  private static Optional<PipelineNameAndParsedParameters> getPipelineNameAndParsedParametersFromSearchFunctions(Map<String, Set<ParameterNameAndLocation>> parsedParameters,
    CallExpression callExpression) {
    return newPipelineNameAndParsedParameters(
      Optional.ofNullable(TreeUtils.nthArgumentOrKeyword(0, "estimator", callExpression.arguments()))
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(RegularArgument.class))
        .map(RegularArgument::expression)
        .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
        .orElse(null),
      parsedParameters);
  }

  private static Optional<PipelineNameAndParsedParameters> newPipelineNameAndParsedParameters(@Nullable Name pipelineName,
    Map<String, Set<ParameterNameAndLocation>> parsedParameters) {
    return Optional.ofNullable(pipelineName)
      .map(pipelineName1 -> new PipelineNameAndParsedParameters(pipelineName1, parsedParameters));
  }

  private static void findProblems(Map<String, Set<ParameterNameAndLocation>> setParameters, Map<String, ClassSymbol> pipelineDefinition, SubscriptionContext subscriptionContext) {
    for (var entry : setParameters.entrySet()) {
      var step = entry.getKey();
      var stringAndTree = entry.getValue();
      var parameters = stringAndTree.stream().map(ParameterNameAndLocation::string).collect(Collectors.toSet());

      var classifier = pipelineDefinition.get(step);
      if (classifier == null) {
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

  private static void createIssue(SubscriptionContext subscriptionContext, String parameter, Set<ParameterNameAndLocation> parameterNameAndLocation) {
    parameterNameAndLocation
      .stream()
      .filter(parameterNameAndLocation1 -> parameterNameAndLocation1.string()
        .equals(parameter))
      .findFirst()
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
      .filter(SklearnPipelineParameterAreCorrectCheck::isTwoElementTuple)
      // If we find a pipeline inside the pipeline, we need to parse it recursively
      .map(SklearnPipelineParameterAreCorrectCheck::createEmptyExpressionAndPrefix)
      .flatMap(expandRecursivePipelines())
      .forEach(
        expressionAndPrefix -> getResult(expressionAndPrefix.tuple())
          .ifPresent(stepAndClassifier1 -> out.put(expressionAndPrefix.prefix() + stepAndClassifier1.stepName(), stepAndClassifier1.classifierName())));
    return out;
  }

  private static boolean isTwoElementTuple(List<Expression> elements) {
    return elements.size() == 2;
  }

  private static ExpressionAndPrefix createEmptyExpressionAndPrefix(List<Expression> tuple) {
    return new ExpressionAndPrefix(tuple, "", 0);
  }

  private static Function<ExpressionAndPrefix, Stream<ExpressionAndPrefix>> expandRecursivePipelines() {
    return expressionAndPrefix -> {
      var tuple = expressionAndPrefix.tuple();
      var step = tuple.get(0);
      var classifier = tuple.get(1);

      if (!step.is(Tree.Kind.STRING_LITERAL) || !classifier.is(Tree.Kind.NAME)) {
        return Stream.of(expressionAndPrefix);
      }
      if (expressionAndPrefix.depth > 10) {
        return Stream.of(expressionAndPrefix);
      }

      return classifierIsANestedPipeline((Name) classifier)
        .map(callExpression -> TreeUtils.nthArgumentOrKeyword(0, "steps", callExpression.arguments()))
        .map(SklearnPipelineParameterAreCorrectCheck::getExpressionsFromArgument)
        .orElse(Stream.empty())
        .map(TreeUtils.toInstanceOfMapper(TupleImpl.class))
        .filter(Objects::nonNull)
        .map(TupleImpl::elements)
        .filter(SklearnPipelineParameterAreCorrectCheck::isTwoElementTuple)
        .map(elements -> new ExpressionAndPrefix(elements,
          expressionAndPrefix.prefix() + ((StringLiteral) step).trimmedQuotesValue() + "__", expressionAndPrefix.depth + 1))
        .flatMap(expandRecursivePipelines());
    };
  }

  private static Optional<CallExpression> classifierIsANestedPipeline(Name classifier) {
    return Expressions.singleAssignedNonNameValue(classifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .filter(callExpression -> Optional.of(callExpression)
        .map(CallExpression::calleeSymbol).map(Symbol::fullyQualifiedName)
        .filter("sklearn.pipeline.Pipeline"::equals)
        .isPresent());
  }

  private static Optional<StepAndClassifier> getResult(List<Expression> tuple) {
    var step = tuple.get(0);
    var classifier = tuple.get(1);
    var stepName = Optional.ofNullable(step)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(StringLiteral.class))
      .map(StringLiteral::trimmedQuotesValue);
    var classifierName = Optional.ofNullable(classifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(CallExpression.class))
      .map(CallExpression::calleeSymbol)
      .filter(symbol -> symbol.is(Symbol.Kind.CLASS) && !"sklearn.pipeline.Pipeline".equals(symbol.fullyQualifiedName()))
      .map(ClassSymbol.class::cast);

    return stepName.flatMap(stepName1 -> classifierName.map(classifierName1 -> new StepAndClassifier(stepName1, classifierName1)));
  }

  private static Map<String, Set<ParameterNameAndLocation>> getStepAndParametersFromArguments(CallExpression callExpression) {
    return callExpression.arguments()
      .stream()
      .filter(argument -> argument.is(Tree.Kind.REGULAR_ARGUMENT))
      .map(RegularArgument.class::cast)
      .map(RegularArgument::keywordArgument)
      .filter(Objects::nonNull)
      .map(SklearnPipelineParameterAreCorrectCheck::getStepAndParameterFromName)
      .<StepAndParameter>mapMulti(Optional::ifPresent)
      .collect(mergeStringAndTreeToMapCollector());
  }

  private static Map<String, Set<ParameterNameAndLocation>> getStepAndParametersFromDict(CallExpression callExpression) {
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

  private static Collector<StepAndParameter, ?, Map<String, Set<ParameterNameAndLocation>>> mergeStringAndTreeToMapCollector() {
    return Collectors.toMap(StepAndParameter::step, stepAndParameter -> Set.of(new ParameterNameAndLocation(stepAndParameter.parameter, stepAndParameter.location)),
      (set1, set2) -> {
        var set = new HashSet<>(set1);
        set.addAll(set2);
        return set;
      });
  }

  private static Optional<StepAndParameter> getStepAndParameterFromName(Name name) {
    return splitStepString(name.name()).map(split -> {
      var splitsNotLast = Arrays.stream(split).limit(split.length - 1L).collect(Collectors.joining("__"));
      return new StepAndParameter(splitsNotLast, split[split.length - 1], name);
    });
  }

  private static Optional<StepAndParameter> getStepAndParameterFromString(String string, Tree location) {
    return splitStepString(string).map(split -> new StepAndParameter(split[0], split[1], location));
  }

  private static Optional<String[]> splitStepString(String string) {
    var split = string.split("__");
    if (split.length < 2 || string.endsWith("__")) {
      return Optional.empty();
    }
    return Optional.of(split);
  }

}
