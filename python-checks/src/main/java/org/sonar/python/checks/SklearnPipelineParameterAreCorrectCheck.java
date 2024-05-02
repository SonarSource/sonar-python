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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.ListLiteralImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.tree.TupleImpl;

@Rule(key = "S6972")
public class SklearnPipelineParameterAreCorrectCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Provide valid parameters to the estimator.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, SklearnPipelineParameterAreCorrectCheck::checkCallExpression);
  }

  private static void checkCallExpression(SubscriptionContext subscriptionContext) {
    CallExpression callExpression = (CallExpression) subscriptionContext.syntaxNode();

    var isRelevantCall = Optional.of(callExpression).map(CallExpression::calleeSymbol).map(Symbol::fullyQualifiedName)
      .filter("sklearn.pipeline.Pipeline.set_params"::equals);

    if (isRelevantCall.isEmpty()) {
      return;
    }

    var parsedFunction = getStepAndParametersFromArguments(callExpression);
    var pipelineName = Optional.of(callExpression).map(CallExpression::callee).flatMap(TreeUtils.toOptionalInstanceOfMapper(QualifiedExpression.class))
      .map(QualifiedExpression::qualifier).flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class));

    pipelineName.flatMap(Expressions::singleAssignedNonNameValue)
      .ifPresent(c1 -> pipelineName.ifPresent(pipelineName1 -> {
        var parsedPipeline = parsePipeline(((CallExpression) c1));
        findProblems(parsedFunction, parsedPipeline, subscriptionContext);
      }));
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
        if (possibleParameters.stream().noneMatch(symbol -> Objects.equals(symbol.name(), parameter))) {
          stringAndTree.stream().filter(stringAndTree1 -> stringAndTree1.string().equals(parameter)).findFirst()
            .ifPresent(location -> subscriptionContext.addIssue(location.tree, MESSAGE));
        }
      });
    }
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
      .filter(Optional::isPresent)
      .map(Optional::get)
      .collect(Collectors.toMap(StepAndParameter::step, stepAndParameter -> Set.of(new StringAndTree(stepAndParameter.parameter(), stepAndParameter.location())),
        (set1, set2) -> {
          var set = new HashSet<>(set1);
          set.addAll(set2);
          return set;
        }));
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

  private record StepAndParameter(String step, String parameter, Tree location) {
  }

}
