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
import org.sonar.plugins.python.api.tree.FunctionDef;
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
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      Optional.ofNullable(functionDef.parameters())
        .ifPresent(parameterList -> removeSelfAndClassParameters(parameterList, (FunctionDefImpl) functionDef).forEach(parameter -> {
          if (parameter.typeAnnotation() == null) {
            ctx.addIssue(parameter, MESSAGE);
          }
        }));
    });
  }

  private Stream<Parameter> removeSelfAndClassParameters(ParameterList parameterList, FunctionDefImpl functionDefImpl) {
    return parameterList.nonTuple().stream()
      .filter(param -> !isASelfInstanceParameter(param, functionDefImpl) && !isAClassMethodParameter(param, functionDefImpl));
  }

  private boolean isASelfInstanceParameter(Parameter parameter, FunctionDefImpl functionDefImpl) {
    return hasName("self", parameter) && isAnInstanceMethod(functionDefImpl);
  }

  private boolean hasName(String name, Parameter parameter) {
    return Optional.ofNullable(parameter.name()).map(parameterName -> name.equals(parameterName.name())).orElse(false);
  }

  private boolean isAnInstanceMethod(FunctionDefImpl functionDefImpl) {
    return Optional.ofNullable(functionDefImpl.functionSymbol())
      .map(FunctionSymbol::isInstanceMethod).orElse(false);
  }

  private boolean isAClassMethodParameter(Parameter parameter, FunctionDefImpl functionDefImpl) {
    return hasName("cls", parameter) && hasAClassMethodDecorator(functionDefImpl);
  }

  private boolean hasAClassMethodDecorator(FunctionDefImpl functionDefImpl) {
    return Optional.ofNullable(functionDefImpl.functionSymbol())
      .map(symbol -> symbol.decorators().contains("classmethod")).orElse(false);
  }
}
