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

import java.util.Optional;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
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
      if (parameterList != null && functionSymbol != null) {
        removeSelfAndClassParameters(parameterList, functionSymbol)
          .forEach(parameter -> {
            if (parameter.typeAnnotation() == null) {
              ctx.addIssue(parameter, MESSAGE);
            }
          });
      }
    });
  }

  private static Stream<Parameter> removeSelfAndClassParameters(ParameterList parameterList, FunctionSymbol functionSymbol) {
    return parameterList.nonTuple().stream()
      .filter(param -> !isASelfInstanceParameter(param, functionSymbol) && !isAClassMethodParameter(param, functionSymbol));
  }

  private static boolean isASelfInstanceParameter(Parameter parameter, FunctionSymbol functionSymbol) {
    return hasName("self", parameter) && functionSymbol.isInstanceMethod();
  }

  private static boolean isAClassMethodParameter(Parameter parameter, FunctionSymbol functionSymbol) {
    return hasName("cls", parameter) && functionSymbol.decorators().contains("classmethod");
  }

  private static boolean hasName(String name, Parameter parameter) {
    return Optional.ofNullable(parameter.name()).map(parameterName -> name.equals(parameterName.name())).orElse(false);
  }
}
