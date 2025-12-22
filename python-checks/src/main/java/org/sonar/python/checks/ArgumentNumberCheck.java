/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S930")
public class ArgumentNumberCheck extends PythonSubscriptionCheck {

  private static final String FUNCTION_DEFINITION = "Function definition.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      if (callExpression.callee().typeV2() instanceof FunctionType functionType) {
        checkFunctionType(ctx, callExpression, functionType);
      }
    });
  }

  private static void checkFunctionType(SubscriptionContext ctx, CallExpression callExpression, FunctionType functionType) {
    if (isException(ctx, callExpression, functionType)) {
      return;
    }
    checkPositionalParameters(ctx, callExpression, functionType);
    checkKeywordArguments(ctx, callExpression, functionType, callExpression.callee());
  }

  private static void checkPositionalParameters(SubscriptionContext ctx, CallExpression callExpression, FunctionType functionType) {
    int self = 0;
    if (isCalledAsInstanceMethod(callExpression, functionType) && functionHasSelfParameter(functionType)) {
      self = 1;
    }
    Map<String, ParameterV2> positionalParamsWithoutDefault = positionalParamsWithoutDefault(functionType);
    long nbPositionalParamsWithDefault = functionType.parameters().stream()
      .filter(parameterName -> !parameterName.isKeywordOnly() && parameterName.hasDefaultValue())
      .count();

    List<RegularArgument> arguments = callExpression.arguments().stream()
      .map(RegularArgument.class::cast)
      .toList();
    long nbPositionalArgs = arguments.stream().filter(a -> a.keywordArgument() == null).count();
    long nbNonKeywordOnlyPassedWithKeyword = arguments.stream()
      .map(RegularArgument::keywordArgument)
      .filter(k -> k != null && positionalParamsWithoutDefault.containsKey(k.name()) && !positionalParamsWithoutDefault.get(k.name()).isPositionalOnly())
      .count();

    int minimumPositionalArgs = positionalParamsWithoutDefault.size();
    String expected = "" + (minimumPositionalArgs - self);
    long nbMissingArgs = minimumPositionalArgs - nbPositionalArgs - self - nbNonKeywordOnlyPassedWithKeyword;
    if (nbMissingArgs > 0) {
      String message = "Add " + nbMissingArgs + " missing arguments; ";
      if (nbPositionalParamsWithDefault > 0) {
        expected = "at least " + expected;
      }
      addPositionalIssue(ctx, callExpression.callee(), functionType, message, expected);
    } else if (nbMissingArgs + nbPositionalParamsWithDefault + nbNonKeywordOnlyPassedWithKeyword < 0) {
      String message = "Remove " + (- nbMissingArgs - nbPositionalParamsWithDefault) + " unexpected arguments; ";
      if (nbPositionalParamsWithDefault > 0) {
        expected = "at most " + (minimumPositionalArgs - self + nbPositionalParamsWithDefault);
      }
      addPositionalIssue(ctx, callExpression.callee(), functionType, message, expected);
    }
  }

  private static boolean isCalledAsInstanceMethod(CallExpression callExpression, FunctionType functionType) {
    return functionType.isInstanceMethod()
      && callExpression.callee() instanceof QualifiedExpression qualifiedExpression
      && qualifiedExpression.qualifier().typeV2() instanceof ObjectType;
  }

  private static boolean functionHasSelfParameter(FunctionType functionType) {
    return !functionType.parameters().isEmpty();
  }

  private static Map<String, ParameterV2> positionalParamsWithoutDefault(FunctionType functionType) {
    int unnamedIndex = 0;
    Map<String, ParameterV2> result = new HashMap<>();
    for (ParameterV2 parameter : functionType.parameters()) {
      if (!parameter.isKeywordOnly() && !parameter.hasDefaultValue()) {
        String name = parameter.name();
        if (name == null || name.isEmpty()) {
          result.put("!unnamed" + unnamedIndex, parameter);
          unnamedIndex++;
        } else {
          result.put(parameter.name(), parameter);
        }
      }
    }
    return result;
  }

  private static void addPositionalIssue(SubscriptionContext ctx, Tree tree, FunctionType functionType, String message, String expected) {
    String msg = message + "'" + functionType.name() + "' expects " + expected + " positional arguments.";
    PreciseIssue preciseIssue = ctx.addIssue(tree, msg);
    addSecondary(functionType, preciseIssue);
  }

  private static boolean isException(SubscriptionContext ctx, CallExpression callExpression, FunctionType functionType) {
    return functionType.hasDecorators()
      || functionType.hasVariadicParameter()
      || callExpression.arguments().stream().anyMatch(argument -> argument.is(Tree.Kind.UNPACKING_EXPR))
      || extendsZopeInterface(ctx, callExpression)
      || isCalledAsBoundInstanceMethod(callExpression)
      || isSuperCall(ctx, callExpression)
      || isReceiverTypeVar(ctx, callExpression)
      || isReceiverTypeInstance(ctx, callExpression);
  }

  private static boolean extendsZopeInterface(SubscriptionContext ctx, CallExpression callExpression) {
    var matcher = TypeMatchers.isFunctionOwnerSatisfying(
      TypeMatchers.isOrExtendsType("zope.interface.Interface")
    );
    return matcher.isTrueFor(callExpression.callee(), ctx);
  }

  private static boolean isCalledAsBoundInstanceMethod(CallExpression callExpression) {
    if (callExpression.callee().typeV2() instanceof FunctionType functionType) {
      return functionType.isInstanceMethod() && !callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR);
    }
    return false;
  }

  private static boolean isReceiverTypeVar(SubscriptionContext ctx, CallExpression callExpression) {
    if (callExpression.callee() instanceof QualifiedExpression qualifiedExpression) {
      return TypeMatchers.isObjectSatisfying(TypeMatchers.isOrExtendsType("typing.TypeVar"))
        .isTrueFor(qualifiedExpression.qualifier(), ctx);
    }
    return false;
  }

  private static boolean isReceiverTypeInstance(SubscriptionContext ctx, CallExpression callExpression) {
    // see SONARPY-3591
    if (callExpression.callee() instanceof QualifiedExpression qualifiedExpression) {
      return TypeMatchers.isObjectSatisfying(TypeMatchers.isOrExtendsType("type"))
        .isTrueFor(qualifiedExpression.qualifier(), ctx);
    }
    return false;
  }

  private static boolean isSuperCall(SubscriptionContext ctx, CallExpression callExpression) {
    if (callExpression.callee() instanceof QualifiedExpression qualifiedExpression) {
      return TypeMatchers.isObjectOfType("super").isTrueFor(qualifiedExpression.qualifier(), ctx);
    }
    return false;
  }

  private static void addSecondary(FunctionType functionType, PreciseIssue preciseIssue) {
    functionType.definitionLocation().ifPresent(location -> preciseIssue.secondary(location, FUNCTION_DEFINITION));
  }

  private static void checkKeywordArguments(SubscriptionContext ctx, CallExpression callExpression, FunctionType functionType, Expression callee) {
    List<ParameterV2> parameters = functionType.parameters();
    Set<String> mandatoryParamNamesKeywordOnly = parameters.stream()
      .filter(parameterName -> parameterName.isKeywordOnly() && !parameterName.hasDefaultValue())
      .map(ParameterV2::name).collect(Collectors.toSet());

    for (Argument argument : callExpression.arguments()) {
      RegularArgument arg = (RegularArgument) argument;
      Name keyword = arg.keywordArgument();
      if (keyword != null) {
        if (parameters.stream().noneMatch(parameter -> keyword.name().equals(parameter.name()) && !parameter.isPositionalOnly())) {
          PreciseIssue preciseIssue = ctx.addIssue(argument, "Remove this unexpected named argument '" + keyword.name() + "'.");
          addSecondary(functionType, preciseIssue);
        } else {
          mandatoryParamNamesKeywordOnly.remove(keyword.name());
        }
      }
    }
    if (!mandatoryParamNamesKeywordOnly.isEmpty()) {
      StringBuilder message = new StringBuilder("Add the missing keyword arguments: ");
      for (String param : mandatoryParamNamesKeywordOnly) {
        message.append("'").append(param).append("' ");
      }
      PreciseIssue preciseIssue = ctx.addIssue(callee, message.toString().trim());
      addSecondary(functionType, preciseIssue);
    }
  }
}
