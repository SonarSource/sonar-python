/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
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
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.semantic.FunctionSymbolImpl;

@Rule(key = "S930")
public class ArgumentNumberCheck extends PythonSubscriptionCheck {

  private static final String FUNCTION_DEFINITION = "Function definition.";

  private static String message(String functionName, long minRequiredPositionalArguments, int nArguments, long nPositionalParamWithDefaultValue) {
    String message = "";
    if (minRequiredPositionalArguments > nArguments) {
      message = "Add " + (minRequiredPositionalArguments - nArguments) + " missing arguments; ";
    } else {
      message = "Remove " + (nArguments - minRequiredPositionalArguments - nPositionalParamWithDefaultValue) + " unexpected arguments; ";
    }
    return message + "'" + functionName + "' expects " + minRequiredPositionalArguments + " positional arguments.";
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CALL_EXPR, ctx -> {
      CallExpression callExpression = (CallExpression) ctx.syntaxNode();
      Symbol symbol = callExpression.calleeSymbol();
      if (symbol != null && symbol.kind() == Symbol.Kind.FUNCTION) {
        FunctionSymbol functionSymbol = (FunctionSymbol) symbol;
        if (isException(callExpression, functionSymbol)) {
          return;
        }
        int nArguments = callExpression.arguments().size();
        int self = 0;
        if (functionSymbol.isInstanceMethod() && callExpression.callee().is(Tree.Kind.QUALIFIED_EXPR)) {
          self = 1;
        }
        long minRequiredPositionalArguments = functionSymbol.parameters().stream()
          .filter(parameterName -> !parameterName.isKeywordOnly() && !parameterName.hasDefaultValue()).count();
        if ((nArguments + self < minRequiredPositionalArguments || nArguments + self > functionSymbol.parameters().size())) {
          PreciseIssue preciseIssue = ctx.addIssue(callExpression.callee(),
            message(functionSymbol.name(), minRequiredPositionalArguments - self, nArguments, functionSymbol.parameters().size() - minRequiredPositionalArguments));
          addSecondary(functionSymbol, preciseIssue);
        }

        checkKeywordArguments(ctx, callExpression, functionSymbol, callExpression.callee());
      }

    });
  }

  private static boolean isException(CallExpression callExpression, FunctionSymbol functionSymbol) {
    return functionSymbol.hasDecorators()
      || functionSymbol.hasVariadicParameter()
      || functionSymbol.isStub()
      || callExpression.arguments().stream().anyMatch(argument -> argument.is(Tree.Kind.UNPACKING_EXPR))
      || extendsZopeInterface(((FunctionSymbolImpl) functionSymbol).owner());
  }

  private static boolean extendsZopeInterface(@Nullable Symbol symbol) {
    if (symbol != null && symbol.kind() == Symbol.Kind.CLASS) {
      Set<Symbol> superClasses = new HashSet<>();
      addSuperClasses(symbol, superClasses);
      return superClasses.stream().anyMatch(s -> "zope.interface.Interface".equals(s.fullyQualifiedName()));
    }
    return false;
  }

  private static void addSuperClasses(Symbol symbol, Set<Symbol> set) {
    if (set.add(symbol) && symbol instanceof ClassSymbol) {
      for (Symbol superClass : ((ClassSymbol) symbol).superClasses()) {
        addSuperClasses(superClass, set);
      }
    }
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
        if (parameters.stream().noneMatch(parameter -> keyword.name().equals(parameter.name()))) {
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
