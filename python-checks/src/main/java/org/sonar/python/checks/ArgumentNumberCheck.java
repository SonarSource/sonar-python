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

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.api.symbols.Usage.Kind.PARAMETER;

@Rule(key = "S930")
public class ArgumentNumberCheck extends PythonSubscriptionCheck {

  private static final String FUNCTION_DEFINITION = "Function definition.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();

      Optional.of(callExpression)
        .map(CallExpression::calleeSymbol)
        .map(SymbolUtils::getFunctionSymbols)
        .filter(SymbolUtils::isEqualParameterCountAndNames)
        .map(Collection::stream)
        .flatMap(Stream::findFirst)
        .ifPresent(functionSymbol -> checkFunctionSymbol(ctx, callExpression, functionSymbol));
    });
  }

  private static void checkFunctionSymbol(SubscriptionContext ctx, CallExpression callExpression, FunctionSymbol functionSymbol) {
    if (isException(callExpression, functionSymbol)) {
      return;
    }
    checkPositionalParameters(ctx, callExpression, functionSymbol);
    checkKeywordArguments(ctx, callExpression, functionSymbol, callExpression.callee());
  }

  private static void checkPositionalParameters(SubscriptionContext ctx, CallExpression callExpression, FunctionSymbol functionSymbol) {
    int self = 0;
    if (functionSymbol.isInstanceMethod() && callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR) && !isCalledAsClassMethod((QualifiedExpression) callExpression.callee())) {
      self = 1;
    }
    Map<String, FunctionSymbol.Parameter> positionalParamsWithoutDefault = positionalParamsWithoutDefault(functionSymbol);
    long nbPositionalParamsWithDefault = functionSymbol.parameters().stream()
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
      addPositionalIssue(ctx, callExpression.callee(), functionSymbol, message, expected);
    } else if (nbMissingArgs + nbPositionalParamsWithDefault + nbNonKeywordOnlyPassedWithKeyword < 0) {
      String message = "Remove " + (- nbMissingArgs - nbPositionalParamsWithDefault) + " unexpected arguments; ";
      if (nbPositionalParamsWithDefault > 0) {
        expected = "at most " + (minimumPositionalArgs - self + nbPositionalParamsWithDefault);
      }
      addPositionalIssue(ctx, callExpression.callee(), functionSymbol, message, expected);
    }
  }

  private static boolean isCalledAsClassMethod(QualifiedExpression callee) {
    return TreeUtils.getSymbolFromTree(callee.qualifier())
      .filter(ArgumentNumberCheck::isParamOfClassMethod)
      .isPresent();
  }

  // no need to check that's the first parameter (i.e. cls)
  // the assumption is that another method can be called only using the first parameter of a class method
  private static boolean isParamOfClassMethod(Symbol symbol) {
    return symbol.usages().stream().anyMatch(usage -> usage.kind() == PARAMETER && isParamOfClassMethod(usage.tree()));
  }

  private static boolean isParamOfClassMethod(Tree tree) {
    FunctionDef functionDef = (FunctionDef) TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF);
    return Optional.ofNullable(TreeUtils.getFunctionSymbolFromDef(functionDef))
      .filter(functionSymbol -> functionSymbol.decorators().stream().anyMatch(dec -> dec.equals("classmethod")))
      .isPresent();
  }

  private static Map<String, FunctionSymbol.Parameter> positionalParamsWithoutDefault(FunctionSymbol functionSymbol) {
    int unnamedIndex = 0;
    Map<String, FunctionSymbol.Parameter> result = new HashMap<>();
    for (FunctionSymbol.Parameter parameter : functionSymbol.parameters()) {
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

  private static void addPositionalIssue(SubscriptionContext ctx, Tree tree, FunctionSymbol functionSymbol, String message, String expected) {
    String msg = message + "'" + functionSymbol.name() + "' expects " + expected + " positional arguments.";
    PreciseIssue preciseIssue = ctx.addIssue(tree, msg);
    addSecondary(functionSymbol, preciseIssue);
  }

  private static boolean isReceiverClassSymbol(QualifiedExpression qualifiedExpression) {
    return TreeUtils.getSymbolFromTree(qualifiedExpression.qualifier())
      .filter(symbol -> symbol.kind() == Symbol.Kind.CLASS)
      .isPresent();
  }

  private static boolean isException(CallExpression callExpression, FunctionSymbol functionSymbol) {
    return functionSymbol.hasDecorators()
      || functionSymbol.hasVariadicParameter()
      || callExpression.arguments().stream().anyMatch(argument -> argument.is(Tree.Kind.UNPACKING_EXPR))
      || extendsZopeInterface(((FunctionSymbolImpl) functionSymbol).owner())
      // TODO: distinguish between class methods (new and old style) from other methods
      || (callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR) && isReceiverClassSymbol(((QualifiedExpression) callExpression.callee())));
  }

  private static boolean extendsZopeInterface(@Nullable Symbol symbol) {
    if (symbol != null && symbol.kind() == Symbol.Kind.CLASS) {
      return ((ClassSymbol) symbol).isOrExtends("zope.interface.Interface");
    }
    return false;
  }

  private static void addSecondary(FunctionSymbol functionSymbol, PreciseIssue preciseIssue) {
    LocationInFile definitionLocation = functionSymbol.definitionLocation();
    if (definitionLocation != null) {
      preciseIssue.secondary(definitionLocation, FUNCTION_DEFINITION);
    }
  }

  private static void checkKeywordArguments(SubscriptionContext ctx, CallExpression callExpression, FunctionSymbol functionSymbol, Expression callee) {
    List<FunctionSymbol.Parameter> parameters = functionSymbol.parameters();
    Set<String> mandatoryParamNamesKeywordOnly = parameters.stream()
      .filter(parameterName -> parameterName.isKeywordOnly() && !parameterName.hasDefaultValue())
      .map(FunctionSymbol.Parameter::name).collect(Collectors.toSet());

    for (Argument argument : callExpression.arguments()) {
      RegularArgument arg = (RegularArgument) argument;
      Name keyword = arg.keywordArgument();
      if (keyword != null) {
        if (parameters.stream().noneMatch(parameter -> keyword.name().equals(parameter.name()) && !parameter.isPositionalOnly())) {
          PreciseIssue preciseIssue = ctx.addIssue(argument, "Remove this unexpected named argument '" + keyword.name() + "'.");
          addSecondary(functionSymbol, preciseIssue);
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
      addSecondary(functionSymbol, preciseIssue);
    }
  }
}
