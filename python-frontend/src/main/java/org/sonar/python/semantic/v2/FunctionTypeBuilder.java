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
package org.sonar.python.semantic.v2;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;

import static org.sonar.python.tree.TreeUtils.locationInFile;

public class FunctionTypeBuilder implements TypeBuilder<FunctionType> {

  private boolean hasVariadicParameter;
  private String name;
  private List<PythonType> attributes;
  private List<ParameterV2> parameters;
  private boolean isAsynchronous;
  private boolean hasDecorators;
  private boolean isInstanceMethod;
  private PythonType owner;
  private PythonType returnType = PythonType.UNKNOWN;

  private static final String CLASS_METHOD_DECORATOR = "classmethod";
  private static final String STATIC_METHOD_DECORATOR = "staticmethod";

  public FunctionTypeBuilder fromFunctionDef(FunctionDef functionDef) {
    this.name = functionDef.name().name();
    this.attributes = new ArrayList<>();
    this.parameters = new ArrayList<>();
    isAsynchronous = functionDef.asyncKeyword() != null;
    hasDecorators = !functionDef.decorators().isEmpty();
    isInstanceMethod = isInstanceMethod(functionDef);
    ParameterList parameterList = functionDef.parameters();
    if (parameterList != null) {
      createParameterNames(parameterList.all(), null);
    }
    return this;
  }

  public FunctionTypeBuilder(String name) {
    this.name = name;
  }

  public FunctionTypeBuilder() {
  }

  public FunctionTypeBuilder withHasVariadicParameter(boolean hasVariadicParameter) {
    this.hasVariadicParameter = hasVariadicParameter;
    return this;
  }

  public FunctionTypeBuilder withAttributes(List<PythonType> attributes) {
    this.attributes = attributes;
    return this;
  }

  public FunctionTypeBuilder withParameters(List<ParameterV2> parameters) {
    this.parameters = parameters;
    return this;
  }

  public FunctionTypeBuilder withAsynchronous(boolean asynchronous) {
    isAsynchronous = asynchronous;
    return this;
  }

  public FunctionTypeBuilder withHasDecorators(boolean hasDecorators) {
    this.hasDecorators = hasDecorators;
    return this;
  }

  public FunctionTypeBuilder withInstanceMethod(boolean instanceMethod) {
    isInstanceMethod = instanceMethod;
    return this;
  }

  public FunctionTypeBuilder withReturnType(PythonType returnType) {
    this.returnType = returnType;
    return this;
  }

  public FunctionType build() {
    return new FunctionType(name, attributes, parameters, returnType, isAsynchronous, hasDecorators, isInstanceMethod, hasVariadicParameter, owner);
  }

  private static boolean isInstanceMethod(FunctionDef functionDef) {
    return !"__new__".equals(functionDef.name().name()) && functionDef.isMethodDefinition() && functionDef.decorators().stream()
      .map(decorator -> TreeUtils.decoratorNameFromExpression(decorator.expression()))
      .filter(Objects::nonNull)
      .noneMatch(decorator -> decorator.equals(STATIC_METHOD_DECORATOR) || decorator.equals(CLASS_METHOD_DECORATOR));
  }

  public FunctionTypeBuilder withOwner(PythonType owner) {
    this.owner = owner;
    return this;
  }

  private void createParameterNames(List<AnyParameter> parameterTrees, @Nullable String fileId) {
    ParameterState parameterState = new ParameterState();
    parameterState.positionalOnly = parameterTrees.stream().anyMatch(param -> Optional.of(param)
      .filter(p -> p.is(Tree.Kind.PARAMETER))
      .map(p -> ((org.sonar.plugins.python.api.tree.Parameter) p).starToken())
      .map(Token::value)
      .filter("/"::equals)
      .isPresent()
    );
    for (AnyParameter anyParameter : parameterTrees) {
      if (anyParameter.is(Tree.Kind.PARAMETER)) {
        addParameter((org.sonar.plugins.python.api.tree.Parameter) anyParameter, fileId, parameterState);
      } else {
        parameters.add(new ParameterV2(null, PythonType.UNKNOWN, false,
          parameterState.keywordOnly, parameterState.positionalOnly, false, false, locationInFile(anyParameter, fileId)));
      }
    }
  }

  private void addParameter(org.sonar.plugins.python.api.tree.Parameter parameter, @Nullable String fileId, ParameterState parameterState) {
    Name parameterName = parameter.name();
    Token starToken = parameter.starToken();
    if (parameterName != null) {
      ParameterType parameterType = getParameterType(parameter);
      var parameterV2 = new ParameterV2(parameterName.name(),
        parameterType.pythonType(),
        parameter.defaultValue() != null,
        parameterState.keywordOnly,
        parameterState.positionalOnly,
        parameterType.isKeywordVariadic(),
        parameterType.isPositionalVariadic(),
        locationInFile(parameter, fileId));

      this.parameters.add(parameterV2);
      if (starToken != null) {
        hasVariadicParameter = true;
        parameterState.keywordOnly = true;
        parameterState.positionalOnly = false;
      }
    } else if (starToken != null) {
      if ("*".equals(starToken.value())) {
        parameterState.keywordOnly = true;
        parameterState.positionalOnly = false;
      }
      if ("/".equals(starToken.value())) {
        parameterState.positionalOnly = false;
      }
    }
  }

  private ParameterType getParameterType(org.sonar.plugins.python.api.tree.Parameter parameter) {
    boolean isPositionalVariadic = false;
    boolean isKeywordVariadic = false;
    Token starToken = parameter.starToken();
    if (starToken != null) {
      // https://docs.python.org/3/reference/compound_stmts.html#function-definitions
      hasVariadicParameter = true;
      if ("*".equals(starToken.value())) {
        // if the form “*identifier” is present, it is initialized to a tuple receiving any excess positional parameters
        isPositionalVariadic = true;
        // Should set PythonType to TUPLE
      }
      if ("**".equals(starToken.value())) {
        //  If the form “**identifier” is present, it is initialized to a new ordered mapping receiving any excess keyword arguments
        isKeywordVariadic = true;
        // Should set PythonType to DICT
      }
    }
    // TODO: SONARPY-1773 handle parameter declared types
    return new ParameterType(PythonType.UNKNOWN, isKeywordVariadic, isPositionalVariadic);
  }

  public static class ParameterState {
    boolean keywordOnly = false;
    boolean positionalOnly = false;
  }

  record ParameterType(PythonType pythonType, boolean isKeywordVariadic, boolean isPositionalVariadic) { }
}
