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
package org.sonar.python.semantic;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.sonar.python.semantic.SymbolUtils.pathOf;
import static org.sonar.python.tree.TreeUtils.locationInFile;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.fromTypeAnnotation;
import static org.sonar.python.types.InferredTypes.fromTypeshedTypeAnnotation;

public class FunctionSymbolImpl extends SymbolImpl implements FunctionSymbol {
  private final List<Parameter> parameters = new ArrayList<>();
  private final List<String> decorators;
  private final LocationInFile functionDefinitionLocation;
  private boolean hasVariadicParameter = false;
  private final boolean isInstanceMethod;
  private final boolean isAsynchronous;
  private final boolean hasDecorators;
  private String annotatedReturnTypeName = null;
  private SymbolsProtos.Type protobufReturnType = null;
  private InferredType declaredReturnType = InferredTypes.anyType();
  private final boolean isStub;
  private Symbol owner;
  private static final String CLASS_METHOD_DECORATOR = "classmethod";
  private static final String STATIC_METHOD_DECORATOR = "staticmethod";
  private boolean isDjangoView = false;
  private boolean hasReadDeclaredReturnType = false;

  FunctionSymbolImpl(FunctionDef functionDef, @Nullable String fullyQualifiedName, PythonFile pythonFile) {
    super(functionDef.name().name(), fullyQualifiedName);
    setKind(Kind.FUNCTION);
    isInstanceMethod = isInstanceMethod(functionDef);
    isAsynchronous = functionDef.asyncKeyword() != null;
    hasDecorators = !functionDef.decorators().isEmpty();
    decorators = decorators(functionDef);
    isStub = false;
    String fileId = Optional.ofNullable(pathOf(pythonFile)).map(Path::toString).orElse(pythonFile.toString());
    functionDefinitionLocation = locationInFile(functionDef.name(), fileId);
  }

  public FunctionSymbolImpl(SymbolsProtos.FunctionSymbol functionSymbolProto, String moduleName) {
    this(functionSymbolProto, null, functionSymbolProto.getValidForList(), moduleName);
  }

  public FunctionSymbolImpl(SymbolsProtos.FunctionSymbol functionSymbolProto, @Nullable String containerClassFqn, String moduleName) {
    this(functionSymbolProto, containerClassFqn, functionSymbolProto.getValidForList(), moduleName);
  }

  public FunctionSymbolImpl(SymbolsProtos.FunctionSymbol functionSymbolProto, @Nullable String containerClassFqn, List<String> validFor, String moduleName) {
    super(functionSymbolProto.getName(), TypeShed.normalizedFqn(functionSymbolProto.getFullyQualifiedName(), moduleName, functionSymbolProto.getName(), containerClassFqn));
    setKind(Kind.FUNCTION);
    isInstanceMethod = containerClassFqn != null && !functionSymbolProto.getIsStatic() && !functionSymbolProto.getIsClassMethod();
    isAsynchronous = functionSymbolProto.getIsAsynchronous();
    hasDecorators = functionSymbolProto.getHasDecorators();
    decorators = functionSymbolProto.getResolvedDecoratorNamesList();
    SymbolsProtos.Type returnAnnotation = functionSymbolProto.getReturnAnnotation();
    String returnTypeName = returnAnnotation.getFullyQualifiedName();
    annotatedReturnTypeName = returnTypeName.isEmpty() ? null : TypeShed.normalizedFqn(returnTypeName);
    protobufReturnType = returnAnnotation;
    for (SymbolsProtos.ParameterSymbol parameterSymbol : functionSymbolProto.getParametersList()) {
      ParameterState parameterState = new ParameterState();
      parameterState.positionalOnly = parameterSymbol.getKind() == SymbolsProtos.ParameterKind.POSITIONAL_ONLY;
      parameterState.keywordOnly = parameterSymbol.getKind() == SymbolsProtos.ParameterKind.KEYWORD_ONLY;
      boolean isKeywordVariadic = parameterSymbol.getKind() == SymbolsProtos.ParameterKind.VAR_KEYWORD;
      boolean isPositionalVariadic = parameterSymbol.getKind() == SymbolsProtos.ParameterKind.VAR_POSITIONAL;
      hasVariadicParameter |= isKeywordVariadic || isPositionalVariadic;
      InferredType declaredType;
      if (isPositionalVariadic) {
        declaredType = InferredTypes.TUPLE;
      } else if (isKeywordVariadic) {
        declaredType = InferredTypes.DICT;
      } else {
        declaredType = anyType();
      }
      String parameterName = parameterSymbol.hasName() ? parameterSymbol.getName() : null;
      ParameterImpl parameter = new ParameterImpl(parameterName, declaredType, null, parameterSymbol.getHasDefault(), parameterState,
        isKeywordVariadic, isPositionalVariadic, parameterSymbol.getTypeAnnotation(), null);
      parameters.add(parameter);
    }
    functionDefinitionLocation = null;
    declaredReturnType = anyType();
    isStub = true;
    isDjangoView = false;
    this.validForPythonVersions = new HashSet<>(validFor);
  }

  public FunctionSymbolImpl(FunctionDescriptor functionDescriptor, String symbolName) {
    super(symbolName, functionDescriptor.fullyQualifiedName());
    setKind(Kind.FUNCTION);
    isInstanceMethod = functionDescriptor.isInstanceMethod();
    isAsynchronous = functionDescriptor.isAsynchronous();
    hasDecorators = functionDescriptor.hasDecorators();
    decorators = functionDescriptor.decorators();
    annotatedReturnTypeName = functionDescriptor.annotatedReturnTypeName();
    functionDefinitionLocation = functionDescriptor.definitionLocation();
    // TODO: Will no longer be true once SONARPY-647 is fixed
    isStub = false;
  }

  public void setParametersWithType(ParameterList parametersList) {
    this.parameters.clear();
    createParameterNames(parametersList.all(), functionDefinitionLocation == null ? null : functionDefinitionLocation.fileId());
  }

  public void addParameter(ParameterImpl parameter) {
    this.parameters.add(parameter);
    if (parameter.isVariadic()) {
      this.hasVariadicParameter = true;
    }
  }

  FunctionSymbolImpl(String name, FunctionSymbol functionSymbol) {
    super(name, functionSymbol.fullyQualifiedName());
    setKind(Kind.FUNCTION);
    isInstanceMethod = functionSymbol.isInstanceMethod();
    isAsynchronous = functionSymbol.isAsynchronous();
    hasDecorators = functionSymbol.hasDecorators();
    decorators = functionSymbol.decorators();
    annotatedReturnTypeName = functionSymbol.annotatedReturnTypeName();
    hasVariadicParameter = functionSymbol.hasVariadicParameter();
    // TODO parameters are shallow
    parameters.addAll(functionSymbol.parameters());
    functionDefinitionLocation = functionSymbol.definitionLocation();
    FunctionSymbolImpl functionSymbolImpl = (FunctionSymbolImpl) functionSymbol;
    protobufReturnType = functionSymbolImpl.protobufReturnType;
    if (functionSymbolImpl.protobufReturnType == null || functionSymbolImpl.hasReadDeclaredReturnType) {
      declaredReturnType = functionSymbolImpl.declaredReturnType();
    }
    isStub = functionSymbol.isStub();
    isDjangoView = functionSymbolImpl.isDjangoView();
    validForPythonVersions = functionSymbolImpl.validForPythonVersions;

  }

  @Override
  public FunctionSymbolImpl copyWithoutUsages() {
    FunctionSymbolImpl copy = new FunctionSymbolImpl(name(), this);
    copy.setKind(kind());
    return copy;
  }

  private static boolean isInstanceMethod(FunctionDef functionDef) {
    return !functionDef.name().name().equals("__new__") && functionDef.isMethodDefinition() && functionDef.decorators().stream()
      .map(decorator -> TreeUtils.decoratorNameFromExpression(decorator.expression()))
      .filter(Objects::nonNull)
      .noneMatch(decorator -> decorator.equals(STATIC_METHOD_DECORATOR) || decorator.equals(CLASS_METHOD_DECORATOR));
  }

  private static List<String> decorators(FunctionDef functionDef) {
    List<String> decoratorNames = new ArrayList<>();
    for (Decorator decorator : functionDef.decorators()) {
      String decoratorName = TreeUtils.decoratorNameFromExpression(decorator.expression());
      if (decoratorName != null) {
        decoratorNames.add(decoratorName);
      }
    }
    return decoratorNames;
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
        parameters.add(new ParameterImpl(null, InferredTypes.anyType(), null, false, parameterState, false, false, null, locationInFile(anyParameter, fileId)));
      }
    }
  }

  private void addParameter(org.sonar.plugins.python.api.tree.Parameter parameter, @Nullable String fileId, ParameterState parameterState) {
    Name parameterName = parameter.name();
    Token starToken = parameter.starToken();
    if (parameterName != null) {
      ParameterType parameterType = getParameterType(parameter);
      this.parameters.add(new ParameterImpl(parameterName.name(), parameterType.inferredType, annotatedTypeName(parameter.typeAnnotation()), parameter.defaultValue() != null,
        parameterState, parameterType.isKeywordVariadic, parameterType.isPositionalVariadic, null, locationInFile(parameter, fileId)));
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
    InferredType inferredType = InferredTypes.anyType();
    boolean isPositionalVariadic = false;
    boolean isKeywordVariadic = false;
    Token starToken = parameter.starToken();
    if (starToken != null) {
      // https://docs.python.org/3/reference/compound_stmts.html#function-definitions
      hasVariadicParameter = true;
      if ("*".equals(starToken.value())) {
        // if the form “*identifier” is present, it is initialized to a tuple receiving any excess positional parameters
        isPositionalVariadic = true;
        inferredType = InferredTypes.TUPLE;
      }
      if ("**".equals(starToken.value())) {
        //  If the form “**identifier” is present, it is initialized to a new ordered mapping receiving any excess keyword arguments
        isKeywordVariadic = true;
        inferredType = InferredTypes.DICT;
      }
    } else {
      TypeAnnotation typeAnnotation = parameter.typeAnnotation();
      if (typeAnnotation != null) {
        inferredType = isStub ? fromTypeshedTypeAnnotation(typeAnnotation) : fromTypeAnnotation(typeAnnotation);
      }
    }
    return new ParameterType(inferredType, isKeywordVariadic, isPositionalVariadic);
  }

  @Override
  public List<String> decorators() {
    return decorators;
  }

  private static class ParameterState {
    boolean keywordOnly = false;
    boolean positionalOnly = false;
  }

  @Override
  public List<Parameter> parameters() {
    return parameters;
  }

  @Override
  public boolean isStub() {
    return isStub;
  }

  @Override
  public boolean isAsynchronous() {
    return isAsynchronous;
  }

  @Override
  public boolean hasVariadicParameter() {
    return hasVariadicParameter;
  }

  @Override
  public boolean isInstanceMethod() {
    return isInstanceMethod;
  }

  @Override
  public boolean hasDecorators() {
    return hasDecorators;
  }

  @Override
  public LocationInFile definitionLocation() {
    return functionDefinitionLocation;
  }

  public InferredType declaredReturnType() {
    if (!hasReadDeclaredReturnType && protobufReturnType != null) {
      declaredReturnType = InferredTypes.fromTypeshedProtobuf(protobufReturnType);
      hasReadDeclaredReturnType = true;
    }
    return declaredReturnType;
  }

  public SymbolsProtos.Type protobufReturnType() {
    return protobufReturnType;
  }
  
  static class ParameterType {
    InferredType inferredType;
    boolean isPositionalVariadic;
    boolean isKeywordVariadic;

    public ParameterType(InferredType inferredType, boolean isKeywordVariadic, boolean isPositionalVariadic) {
      this.inferredType = inferredType;
      this.isKeywordVariadic = isKeywordVariadic;
      this.isPositionalVariadic = isPositionalVariadic;
    }
  }

  private String annotatedTypeName(@Nullable TypeAnnotation typeAnnotation) {
    return Optional.ofNullable(typeAnnotation)
      .map(TypeAnnotation::expression)
      .map(SymbolImpl::getTypeSymbolFromExpression)
      .map(Symbol::fullyQualifiedName)
      .orElse(null);
  }

  @Override
  public String annotatedReturnTypeName() {
    return annotatedReturnTypeName;
  }

  public void setDeclaredReturnType(InferredType declaredReturnType) {
    this.declaredReturnType = declaredReturnType;
  }

  public Symbol owner() {
    return owner;
  }

  public void setOwner(Symbol owner) {
    this.owner = owner;
  }

  public boolean isDjangoView() {
    return isDjangoView;
  }

  public void setIsDjangoView(boolean isDjangoView) {
    this.isDjangoView = isDjangoView;
  }

  public static class ParameterImpl implements Parameter {

    private final String name;
    private InferredType declaredType;
    private SymbolsProtos.Type protobufType;
    private final String annotatedTypeName;
    private final boolean hasDefaultValue;
    private final boolean isKeywordVariadic;
    private final boolean isPositionalVariadic;
    private final boolean isKeywordOnly;
    private final boolean isPositionalOnly;
    private final LocationInFile location;
    private boolean hasReadDeclaredType = false;

    ParameterImpl(@Nullable String name, InferredType declaredType, @Nullable String annotatedTypeName, boolean hasDefaultValue,
      ParameterState parameterState, boolean isKeywordVariadic, boolean isPositionalVariadic, @Nullable SymbolsProtos.Type protobufType, @Nullable LocationInFile location) {
      this.name = name;
      this.declaredType = declaredType;
      this.hasDefaultValue = hasDefaultValue;
      this.isKeywordVariadic = isKeywordVariadic;
      this.isPositionalVariadic = isPositionalVariadic;
      this.isKeywordOnly = parameterState.keywordOnly;
      this.isPositionalOnly = parameterState.positionalOnly;
      this.location = location;
      this.protobufType = protobufType;
      this.annotatedTypeName = annotatedTypeName;
    }

    public ParameterImpl(FunctionDescriptor.Parameter parameterDescriptor) {
      this.name = parameterDescriptor.name();
      this.hasDefaultValue = parameterDescriptor.hasDefaultValue();
      this.isPositionalVariadic = parameterDescriptor.isPositionalVariadic();
      this.isKeywordVariadic = parameterDescriptor.isKeywordVariadic();
      this.isKeywordOnly = parameterDescriptor.isKeywordOnly();
      this.isPositionalOnly = parameterDescriptor.isPositionalOnly();
      this.location = parameterDescriptor.location();
      this.annotatedTypeName = parameterDescriptor.annotatedType();
    }

    @Override
    @CheckForNull
    public String name() {
      return name;
    }

    @Override
    public InferredType declaredType() {
      if (!hasReadDeclaredType && protobufType != null) {
        declaredType = InferredTypes.fromTypeshedProtobuf(protobufType);
        hasReadDeclaredType = true;
      }
      return declaredType;
    }

    public void setDeclaredType(InferredType type) {
      this.declaredType = type;
    }

    @CheckForNull
    public String annotatedTypeName() {
      return annotatedTypeName;
    }

    @Override
    public boolean hasDefaultValue() {
      return hasDefaultValue;
    }

    @Override
    public boolean isVariadic() {
      return isKeywordVariadic || isPositionalVariadic;
    }

    @Override
    public boolean isKeywordOnly() {
      return isKeywordOnly;
    }

    @Override
    public boolean isPositionalOnly() {
      return isPositionalOnly;
    }

    @Override
    public boolean isKeywordVariadic() {
      return isKeywordVariadic;
    }

    @Override
    public boolean isPositionalVariadic() {
      return isPositionalVariadic;
    }

    @CheckForNull
    @Override
    public LocationInFile location() {
      return location;
    }
  }
}
