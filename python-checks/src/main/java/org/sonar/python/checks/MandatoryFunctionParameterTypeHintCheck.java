/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.FunctionDefImpl;

@Rule(key = "S6540")
public class MandatoryFunctionParameterTypeHintCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add a type hint to this function parameter.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDefImpl functionDef = (FunctionDefImpl) ctx.syntaxNode();
      ParameterList parameterList = functionDef.parameters();
      FunctionSymbol functionSymbol = functionDef.functionSymbol();
      if (parameterList != null) {
        List<Parameter> parameters = parameterList.nonTuple();
        for (int i = 0; i < parameters.size(); i++) {
          Parameter parameter = parameters.get(i);
          boolean isFirstParameter = i == 0;
          if (!isASelfInstanceParameter(parameter, functionSymbol) &&
            !isFirstParamOfClassAnnotatedMethod(functionSymbol, isFirstParameter) &&
            !isFirstParamOfNewMethod(functionSymbol, isFirstParameter) &&
            !isSpecialCharParameter(parameter) && parameter.typeAnnotation() == null) {
            ctx.addIssue(parameter, MESSAGE);
          }
        }
      }
    });
  }

  private static boolean isASelfInstanceParameter(Parameter parameter, @Nullable FunctionSymbol functionSymbol) {
    return hasName("self", parameter) && functionSymbol != null && functionSymbol.isInstanceMethod();
  }

  private static boolean isFirstParamOfClassAnnotatedMethod(@Nullable FunctionSymbol functionSymbol, boolean isFirstParameter) {
    return functionSymbol != null && functionSymbol.decorators().contains("classmethod") && isFirstParameter;
  }

  private static boolean isFirstParamOfNewMethod(@Nullable FunctionSymbol functionSymbol, boolean isFirstParameter) {
    return functionSymbol != null && "__new__".equals(functionSymbol.name()) && isFirstParameter;
  }

  private static final List<String> specialTokenParams = Arrays.asList("*", "/");

  private static boolean isSpecialCharParameter(Parameter parameter) {
    Name parameterName = parameter.name();
    Token token = parameter.starToken();
    return (hasName("_", parameter)) ||
      (parameterName == null && token != null && specialTokenParams.contains(token.value()));
  }

  private static boolean hasName(String name, Parameter parameter) {
    return Optional.ofNullable(parameter.name()).map(parameterName -> name.equals(parameterName.name())).orElse(false);
  }
}
