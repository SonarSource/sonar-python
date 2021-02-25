/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.semantic;

import java.nio.file.Path;
import java.util.ArrayList;
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
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.semantic.SymbolUtils.isTypeShedFile;
import static org.sonar.python.semantic.SymbolUtils.pathOf;
import static org.sonar.python.tree.TreeUtils.locationInFile;
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
  private InferredType declaredReturnType = InferredTypes.anyType();
  private final boolean isStub;
  private Symbol owner;
  private static final String CLASS_METHOD_DECORATOR = "classmethod";
  private static final String STATIC_METHOD_DECORATOR = "staticmethod";
  private boolean isDjangoView = false;

  FunctionSymbolImpl(FunctionDef functionDef, @Nullable String fullyQualifiedName, PythonFile pythonFile) {
    super(functionDef.name().name(), fullyQualifiedName);
    setKind(Kind.FUNCTION);
    isInstanceMethod = isInstanceMethod(functionDef);
    isAsynchronous = functionDef.asyncKeyword() != null;
    hasDecorators = !functionDef.decorators().isEmpty();
    decorators = decorators(functionDef);
    String fileId = null;
    isStub = isTypeShedFile(pythonFile);
    if (!isStub) {
      Path path = pathOf(pythonFile);
      fileId = path != null ? path.toString() : pythonFile.toString();
    }
    functionDefinitionLocation = locationInFile(functionDef.name(), fileId);
  }

  public void setParametersWithType(ParameterList parametersList) {
    this.parameters.clear();
    createParameterNames(parametersList.all(), functionDefinitionLocation == null ? null : functionDefinitionLocation.fileId());
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
    parameters.addAll(functionSymbol.parameters());
    functionDefinitionLocation = functionSymbol.definitionLocation();
    declaredReturnType = ((FunctionSymbolImpl) functionSymbol).declaredReturnType();
    isStub = functionSymbol.isStub();
    isDjangoView = ((FunctionSymbolImpl) functionSymbol).isDjangoView();
  }

  @Override
  FunctionSymbolImpl copyWithoutUsages() {
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
        parameters.add(new ParameterImpl(null, InferredTypes.anyType(), false, false, parameterState, locationInFile(anyParameter, fileId)));
      }
    }
  }

  private void addParameter(org.sonar.plugins.python.api.tree.Parameter parameter, @Nullable String fileId, ParameterState parameterState) {
    Name parameterName = parameter.name();
    Token starToken = parameter.starToken();
    if (parameterName != null) {
      InferredType declaredType = getParameterType(parameter, starToken);
      this.parameters.add(new ParameterImpl(parameterName.name(), declaredType, parameter.defaultValue() != null,
        starToken != null, parameterState, locationInFile(parameter, fileId)));
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

  private InferredType getParameterType(org.sonar.plugins.python.api.tree.Parameter parameter, @Nullable Token starToken) {
    if (starToken != null) {
      // https://docs.python.org/3/reference/compound_stmts.html#function-definitions
      if ("*".equals(starToken.value())) {
        // if the form “*identifier” is present, it is initialized to a tuple receiving any excess positional parameters
        return InferredTypes.TUPLE;
      }
      if ("**".equals(starToken.value())) {
        //  If the form “**identifier” is present, it is initialized to a new ordered mapping receiving any excess keyword arguments
        return InferredTypes.DICT;
      }
    }
    InferredType declaredType = InferredTypes.anyType();
    TypeAnnotation typeAnnotation = parameter.typeAnnotation();
    if (typeAnnotation != null) {
      declaredType = isStub ? fromTypeshedTypeAnnotation(typeAnnotation) : fromTypeAnnotation(typeAnnotation);
    }
    return declaredType;
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
    return declaredReturnType;
  }

  public void setAnnotatedReturnTypeName(@Nullable TypeAnnotation returnTypeAnnotation) {
    annotatedReturnTypeName = Optional.ofNullable(returnTypeAnnotation)
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

  private static class ParameterImpl implements Parameter {

    private final String name;
    private final InferredType declaredType;
    private final boolean hasDefaultValue;
    private final boolean isVariadic;
    private final boolean isKeywordOnly;
    private final boolean isPositionalOnly;
    private final LocationInFile location;

    ParameterImpl(@Nullable String name, InferredType declaredType, boolean hasDefaultValue,
                  boolean isVariadic, ParameterState parameterState, @Nullable LocationInFile location) {
      this.name = name;
      this.declaredType = declaredType;
      this.hasDefaultValue = hasDefaultValue;
      this.isVariadic = isVariadic;
      this.isKeywordOnly = parameterState.keywordOnly;
      this.isPositionalOnly = parameterState.positionalOnly;
      this.location = location;
    }

    @Override
    @CheckForNull
    public String name() {
      return name;
    }

    @Override
    public InferredType declaredType() {
      return declaredType;
    }

    @Override
    public boolean hasDefaultValue() {
      return hasDefaultValue;
    }

    @Override
    public boolean isVariadic() {
      return isVariadic;
    }

    @Override
    public boolean isKeywordOnly() {
      return isKeywordOnly;
    }

    @Override
    public boolean isPositionalOnly() {
      return isPositionalOnly;
    }

    @CheckForNull
    @Override
    public LocationInFile location() {
      return location;
    }
  }
}
