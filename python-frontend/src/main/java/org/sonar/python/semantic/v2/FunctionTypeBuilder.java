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
package org.sonar.python.semantic.v2;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.SimpleTypeWrapper;
import org.sonar.python.types.v2.TypeOrigin;
import org.sonar.python.types.v2.TypeWrapper;

import static org.sonar.python.tree.TreeUtils.locationInFile;

public class FunctionTypeBuilder implements TypeBuilder<FunctionType> {

  private boolean hasVariadicParameter;
  private String name;
  private String fullyQualifiedName;
  private List<PythonType> attributes;
  private List<ParameterV2> parameters;
  private List<TypeWrapper> decorators;
  private boolean isAsynchronous;
  private boolean hasDecorators;
  private boolean isInstanceMethod;
  private PythonType owner;
  private TypeWrapper returnType = TypeWrapper.UNKNOWN_TYPE_WRAPPER;
  private TypeOrigin typeOrigin = TypeOrigin.STUB;
  private LocationInFile definitionLocation;

  private static final String CLASS_METHOD_DECORATOR = "classmethod";
  private static final String STATIC_METHOD_DECORATOR = "staticmethod";

  public FunctionTypeBuilder fromFunctionDef(FunctionDef functionDef, String fullyQualifiedName, @Nullable String fileId, TypeTable projectLevelTypeTable) {
    this.name = functionDef.name().name();
    this.fullyQualifiedName = fullyQualifiedName;
    this.attributes = new ArrayList<>();
    this.parameters = new ArrayList<>();
    isAsynchronous = functionDef.asyncKeyword() != null;
    hasDecorators = !functionDef.decorators().isEmpty();
    this.decorators = functionDef.decorators()
      .stream()
      .map(Decorator::expression)
      .map(Expression::typeV2)
      .map(TypeWrapper::of)
      .toList();
    isInstanceMethod = isInstanceMethod(functionDef);
    ParameterList parameterList = functionDef.parameters();
    if (parameterList != null) {
      createParameterNames(parameterList.all(), fileId,  projectLevelTypeTable);
    }
    return this;
  }

  public FunctionTypeBuilder(String name) {
    this.name = name;
  }

  public FunctionTypeBuilder() {
  }

  public FunctionTypeBuilder withName(String name) {
    this.name = name;
    return this;
  }

  public FunctionTypeBuilder withFullyQualifiedName(@Nullable String fullyQualifiedName) {
    this.fullyQualifiedName = fullyQualifiedName;
    return this;
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

  public FunctionTypeBuilder withDecorators(List<TypeWrapper> decorators) {
    this.decorators = decorators;
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
    withReturnType(new LazyTypeWrapper(returnType));
    return this;
  }

  public FunctionTypeBuilder withReturnType(TypeWrapper returnType) {
    this.returnType = returnType;
    return this;
  }

  public FunctionTypeBuilder withTypeOrigin(TypeOrigin typeOrigin) {
    this.typeOrigin = typeOrigin;
    return this;
  }

  @Override
  public FunctionTypeBuilder withDefinitionLocation(@Nullable LocationInFile definitionLocation) {
    this.definitionLocation = definitionLocation;
    return this;
  }

  public FunctionType build() {
    return new FunctionType(
      name, fullyQualifiedName, attributes, parameters, decorators, returnType, typeOrigin,
      isAsynchronous, hasDecorators, isInstanceMethod, hasVariadicParameter, owner, definitionLocation
    );
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

  private void createParameterNames(List<AnyParameter> parameterTrees, @Nullable String fileId, TypeTable projectLevelTypeTable) {
    ParameterState parameterState = new ParameterState();
    parameterState.positionalOnly = parameterTrees.stream().anyMatch(param -> Optional.of(param)
      .filter(p -> p.is(Tree.Kind.PARAMETER))
      .map(p -> ((Parameter) p).starToken())
      .map(Token::value)
      .filter("/"::equals)
      .isPresent()
    );
    for (AnyParameter anyParameter : parameterTrees) {
      if (anyParameter.is(Tree.Kind.PARAMETER)) {
        addParameter((Parameter) anyParameter, fileId, parameterState, projectLevelTypeTable);
      } else {
        parameters.add(new ParameterV2(null, new SimpleTypeWrapper(PythonType.UNKNOWN), false,
          parameterState.keywordOnly, parameterState.positionalOnly, false, false, locationInFile(anyParameter, fileId)));
      }
    }
  }

  private void addParameter(Parameter parameter, @Nullable String fileId, ParameterState parameterState, TypeTable projectLevelTypeTable) {
    Name parameterName = parameter.name();
    Token starToken = parameter.starToken();
    if (parameterName != null) {
      ParameterType parameterType = getParameterType(parameter, projectLevelTypeTable);
      this.parameters.add(new ParameterV2(parameterName.name(), new LazyTypeWrapper(parameterType.pythonType()), parameter.defaultValue() != null,
        parameterState.keywordOnly, parameterState.positionalOnly, parameterType.isKeywordVariadic(), parameterType.isPositionalVariadic(), locationInFile(parameter, fileId)));
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

  private ParameterType getParameterType(Parameter parameter, TypeTable projectLevelTypeTable) {
    boolean isPositionalVariadic = false;
    boolean isKeywordVariadic = false;
    Token starToken = parameter.starToken();
    var parameterType = Optional.ofNullable(parameter.name()).map(Name::typeV2).orElse(PythonType.UNKNOWN);
    if (starToken != null) {
      // https://docs.python.org/3/reference/compound_stmts.html#function-definitions
      hasVariadicParameter = true;
      if ("*".equals(starToken.value())) {
        // if the form “*identifier” is present, it is initialized to a tuple receiving any excess positional parameters
        isPositionalVariadic = true;
        // Should set PythonType to TUPLE
        parameterType = projectLevelTypeTable.getBuiltinsModule().resolveMember("tuple").orElse(PythonType.UNKNOWN);
      }
      if ("**".equals(starToken.value())) {
        //  If the form “**identifier” is present, it is initialized to a new ordered mapping receiving any excess keyword arguments
        isKeywordVariadic = true;
        // Should set PythonType to DICT
        parameterType = projectLevelTypeTable.getBuiltinsModule().resolveMember("dict").orElse(PythonType.UNKNOWN);
      }
    }
    return new ParameterType(parameterType, isKeywordVariadic, isPositionalVariadic);
  }

  public static class ParameterState {
    boolean keywordOnly = false;
    boolean positionalOnly = false;
  }

  record ParameterType(PythonType pythonType, boolean isKeywordVariadic, boolean isPositionalVariadic) { }
}
