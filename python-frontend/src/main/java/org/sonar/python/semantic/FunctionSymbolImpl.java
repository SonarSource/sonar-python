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
package org.sonar.python.semantic;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
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
import org.sonar.python.TokenLocation;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.semantic.SymbolUtils.pathOf;

public class FunctionSymbolImpl extends SymbolImpl implements FunctionSymbol {
  private final List<Parameter> parameters = new ArrayList<>();
  private final List<String> decorators;
  private final LocationInFile functionDefinitionLocation;
  private boolean hasVariadicParameter = false;
  private final boolean isInstanceMethod;
  private final boolean hasDecorators;
  private InferredType declaredReturnType = InferredTypes.anyType();
  private boolean isStub = false;
  private Symbol owner;
  private static final String CLASS_METHOD_DECORATOR = "classmethod";
  private static final String STATIC_METHOD_DECORATOR = "staticmethod";

  FunctionSymbolImpl(FunctionDef functionDef, @Nullable String fullyQualifiedName, PythonFile pythonFile) {
    super(functionDef.name().name(), fullyQualifiedName);
    setKind(Kind.FUNCTION);
    isInstanceMethod = isInstanceMethod(functionDef);
    hasDecorators = !functionDef.decorators().isEmpty();
    decorators = decorators(functionDef);
    String fileId = null;
    if (!SymbolUtils.isTypeShedFile(pythonFile)) {
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
    hasDecorators = functionSymbol.hasDecorators();
    decorators = functionSymbol.decorators();
    hasVariadicParameter = functionSymbol.hasVariadicParameter();
    parameters.addAll(functionSymbol.parameters());
    functionDefinitionLocation = functionSymbol.definitionLocation();
    declaredReturnType = ((FunctionSymbolImpl) functionSymbol).declaredReturnType();
    isStub = functionSymbol.isStub();
  }

  public FunctionSymbolImpl(String name, @Nullable String fullyQualifiedName, boolean hasVariadicParameter,
                            boolean isInstanceMethod, boolean hasDecorators, List<Parameter> parameters, List<String> decorators) {
    super(name, fullyQualifiedName);
    setKind(Kind.FUNCTION);
    this.hasVariadicParameter = hasVariadicParameter;
    this.isInstanceMethod = isInstanceMethod;
    this.hasDecorators = hasDecorators;
    this.decorators = decorators;
    this.parameters.addAll(parameters);
    this.functionDefinitionLocation = null;
    this.isStub = true;
  }

  @CheckForNull
  private static LocationInFile locationInFile(Tree tree, @Nullable String fileId) {
    if (fileId == null) {
      return null;
    }
    TokenLocation firstToken = new TokenLocation(tree.firstToken());
    TokenLocation lastToken = new TokenLocation(tree.lastToken());
    return new LocationInFile(fileId, firstToken.startLine(), firstToken.startLineOffset(), lastToken.endLine(), lastToken.endLineOffset());
  }

  @Override
  FunctionSymbolImpl copyWithoutUsages() {
    FunctionSymbolImpl copy = new FunctionSymbolImpl(name(), this);
    copy.setKind(kind());
    return copy;
  }

  private static boolean isInstanceMethod(FunctionDef functionDef) {
    return !functionDef.name().name().equals("__new__") && functionDef.isMethodDefinition() && functionDef.decorators().stream()
      .map(decorator -> {
        List<Name> names = decorator.name().names();
        return names.get(names.size() - 1).name();
      })
      .noneMatch(decorator -> decorator.equals(STATIC_METHOD_DECORATOR) || decorator.equals(CLASS_METHOD_DECORATOR));
  }

  private static List<String> decorators(FunctionDef functionDef) {
    List<String> decoratorNames = new ArrayList<>();
    for (Decorator decorator : functionDef.decorators()) {
      String name = decorator.name().names().stream().map(Name::name).collect(Collectors.joining("."));
      decoratorNames.add(name);
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
      TypeAnnotation typeAnnotation = parameter.typeAnnotation();
      InferredType declaredType = InferredTypes.anyType();
      if (typeAnnotation != null) {
        declaredType = InferredTypes.declaredType(typeAnnotation);
      }
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

  public void setDeclaredReturnType(InferredType declaredReturnType) {
    this.declaredReturnType = declaredReturnType;
  }

  public Symbol owner() {
    return owner;
  }

  public void setOwner(Symbol owner) {
    this.owner = owner;
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
