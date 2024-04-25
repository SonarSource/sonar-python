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
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.UsageV2;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.UnionType;

@Rule(key = "S930")
public class ArgumentNumberCheck extends PythonSubscriptionCheck {

  private static final String FUNCTION_DEFINITION = "Function definition.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      Optional.of(callExpression)
        .map(CallExpression::callee)
        .map(Expression::typeV2)
        .flatMap(ArgumentNumberCheck::getFunctionType)
        .ifPresent(functionType -> checkFunctionType(ctx, callExpression, functionType));
    });
  }

  private static Optional<FunctionType> getFunctionType(PythonType pythonType) {
    if (pythonType instanceof FunctionType functionType) {
      return Optional.of(functionType);
    } else if (pythonType instanceof UnionType unionType) {
      var functionTypesOnly = unionType.candidates()
        .stream()
        .allMatch(FunctionType.class::isInstance);

      if (functionTypesOnly) {
        var sameAmountOfParams = unionType.candidates()
          .stream()
          .map(FunctionType.class::cast)
          .map(FunctionType::parameters)
          .map(List::size)
          .distinct()
          .count() == 1;

        if (sameAmountOfParams) {
          return unionType.candidates()
            .stream()
            .map(FunctionType.class::cast)
            .findFirst();
        }
      }
    }
    return Optional.empty();
  }

  private static void checkFunctionType(SubscriptionContext ctx, CallExpression callExpression, FunctionType functionType) {
    if (isException(callExpression, functionType)) {
      return;
    }
    checkPositionalParameters(ctx, callExpression, functionType);
    checkKeywordArguments(ctx, callExpression, functionType, callExpression.callee());
  }

  private static void checkPositionalParameters(SubscriptionContext ctx, CallExpression callExpression, FunctionType functionType) {
    var self = 0;
    if (functionType.isInstanceMethod() && callExpression.callee() instanceof QualifiedExpression callee && !isCalledAsClassMethod(callee)) {
      self = 1;
    }
    var positionalParamsWithoutDefault = positionalParamsWithoutDefault(functionType);
    var nbPositionalParamsWithDefault = functionType.parameters()
      .stream()
      .filter(Predicate.not(ParameterV2::isKeywordOnly))
      .filter(ParameterV2::hasDefaultValue)
      .count();

    var arguments = callExpression.arguments().stream()
      .map(RegularArgument.class::cast)
      .toList();

    var nbPositionalArgs = arguments.stream()
      .map(RegularArgument::keywordArgument)
      .filter(Objects::isNull)
      .count();

    var nbNonKeywordOnlyPassedWithKeyword = arguments.stream()
      .map(RegularArgument::keywordArgument)
      .filter(Objects::nonNull)
      .map(Name::name)
      .filter(positionalParamsWithoutDefault::containsKey)
      .map(positionalParamsWithoutDefault::get)
      .filter(Predicate.not(ParameterV2::isPositionalOnly))
      .count();

    var minimumPositionalArgs = positionalParamsWithoutDefault.size();
    var expected = "" + (minimumPositionalArgs - self);
    var nbMissingArgs = minimumPositionalArgs - nbPositionalArgs - self - nbNonKeywordOnlyPassedWithKeyword;
    if (nbMissingArgs > 0) {
      var message = "Add " + nbMissingArgs + " missing arguments; ";
      if (nbPositionalParamsWithDefault > 0) {
        expected = "at least " + expected;
      }
      addPositionalIssue(ctx, callExpression.callee(), functionType, message, expected);
    } else if (nbMissingArgs + nbPositionalParamsWithDefault + nbNonKeywordOnlyPassedWithKeyword < 0) {
      var message = "Remove " + (-nbMissingArgs - nbPositionalParamsWithDefault) + " unexpected arguments; ";
      if (nbPositionalParamsWithDefault > 0) {
        expected = "at most " + (minimumPositionalArgs - self + nbPositionalParamsWithDefault);
      }
      addPositionalIssue(ctx, callExpression.callee(), functionType, message, expected);
    }
  }

  private static void addPositionalIssue(SubscriptionContext ctx, Tree callee, FunctionType functionType, String message,
    String expected) {
    String msg = message + "'" + functionType.name() + "' expects " + expected + " positional arguments.";
    PreciseIssue preciseIssue = ctx.addIssue(callee, msg);
    addSecondary(callee, preciseIssue);
  }

  private static void checkKeywordArguments(SubscriptionContext ctx, CallExpression callExpression, FunctionType functionType,
    Expression callee) {
    var parameters = functionType.parameters();
    var mandatoryParamNamesKeywordOnly = parameters.stream()
      .filter(ParameterV2::isKeywordOnly)
      .filter(Predicate.not(ParameterV2::hasDefaultValue))
      .map(ParameterV2::name)
      .collect(Collectors.toSet());

    for (var argument : callExpression.arguments()) {
      var arg = (RegularArgument) argument;
      var keyword = arg.keywordArgument();
      if (keyword != null) {
        if (parameters.stream().noneMatch(parameter -> keyword.name().equals(parameter.name()) && !parameter.isPositionalOnly())) {
          var preciseIssue = ctx.addIssue(argument, "Remove this unexpected named argument '" + keyword.name() + "'.");
          addSecondary(callExpression.callee(), preciseIssue);
        } else {
          mandatoryParamNamesKeywordOnly.remove(keyword.name());
        }
      }
    }
    if (!mandatoryParamNamesKeywordOnly.isEmpty()) {
      var message = new StringBuilder("Add the missing keyword arguments: ");
      for (var param : mandatoryParamNamesKeywordOnly) {
        message.append("'").append(param).append("' ");
      }
      var preciseIssue = ctx.addIssue(callee, message.toString().trim());
      addSecondary(callExpression.callee(), preciseIssue);
    }
  }

  private static void addSecondary(Tree callee, PreciseIssue preciseIssue) {
    Optional.of(callee)
      .filter(Name.class::isInstance)
      .map(Name.class::cast)
      .map(Name::symbolV2)
      .map(SymbolV2::usages)
      .stream()
      .flatMap(Collection::stream)
      .filter(UsageV2::isBindingUsage)
      .findFirst()
      .map(UsageV2::tree)
      .ifPresent(tree -> preciseIssue.secondary(tree, FUNCTION_DEFINITION));
  }

  private static Map<String, ParameterV2> positionalParamsWithoutDefault(FunctionType functionType) {
    int unnamedIndex = 0;
    var result = new HashMap<String, ParameterV2>();
    for (var parameter : functionType.parameters()) {
      if (!parameter.isKeywordOnly() && !parameter.hasDefaultValue()) {
        String name = parameter.name();
        if (name == null) {
          result.put("!unnamed" + unnamedIndex, parameter);
          unnamedIndex++;
        } else {
          result.put(parameter.name(), parameter);
        }
      }
    }
    return result;
  }

  private static boolean isException(CallExpression callExpression, FunctionType functionType) {
    return functionType.hasDecorators()
      || functionType.hasVariadicParameter()
      || callExpression.arguments().stream().anyMatch(argument -> argument.is(Tree.Kind.UNPACKING_EXPR))
      || extendsZopeInterface(functionType.owner())
      // TODO: distinguish between class methods (new and old style) from other methods
      || (callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR) && isReceiverClassSymbol(((QualifiedExpression) callExpression.callee())))
      || (functionType.isInstanceMethod() && !callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR));
  }

  private static boolean extendsZopeInterface(@Nullable PythonType type) {
    if (type instanceof ClassType classType) {
      return classType.isOrExtends("zope.interface.Interface");
    }
    return false;
  }

  private static boolean isCalledAsClassMethod(QualifiedExpression callee) {
    return Optional.of(callee)
      .map(QualifiedExpression::qualifier)
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(Name.class))
      .map(Name::symbolV2)
      .map(SymbolV2::usages)
      .stream()
      .flatMap(Collection::stream)
      .anyMatch(usage -> usage.kind() == UsageV2.Kind.PARAMETER && isParamOfClassMethod(usage.tree()));
  }

  private static boolean isParamOfClassMethod(Tree tree) {
    return Optional.of(tree)
      .map(t -> TreeUtils.firstAncestorOfKind(t, Tree.Kind.FUNCDEF))
      .flatMap(TreeUtils.toOptionalInstanceOfMapper(FunctionDef.class))
      .map(FunctionDef::decorators)
      .stream()
      .flatMap(Collection::stream)
      .map(Decorator::expression)
      .map(Expression::typeV2)
      .filter(ClassType.class::isInstance)
      .map(ClassType.class::cast)
      .anyMatch(classType -> "classtype".equals(classType.name()));
  }

  private static boolean isReceiverClassSymbol(QualifiedExpression qualifiedExpression) {
    return qualifiedExpression.qualifier().typeV2() instanceof ClassType;
  }


}
