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
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.TokenLocation;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.semantic.SymbolUtils.pathOf;

public class FunctionSymbolImpl extends SymbolImpl implements FunctionSymbol {
  private final List<Parameter> parameters = new ArrayList<>();
  private final LocationInFile functionDefinitionLocation;
  private boolean hasVariadicParameter = false;
  private final boolean isInstanceMethod;
  private final boolean hasDecorators;
  private InferredType declaredReturnType = InferredTypes.anyType();
  private boolean isStub = false;
  private Symbol owner;

  FunctionSymbolImpl(FunctionDef functionDef, @Nullable String fullyQualifiedName, PythonFile pythonFile) {
    super(functionDef.name().name(), fullyQualifiedName);
    setKind(Kind.FUNCTION);
    isInstanceMethod = isInstanceMethod(functionDef);
    hasDecorators = !functionDef.decorators().isEmpty();
    String fileId = null;
    if (!SymbolUtils.isTypeShedFile(pythonFile)) {
      Path path = pathOf(pythonFile);
      fileId = path != null ? path.toString() : pythonFile.toString();
    }
    ParameterList parametersList = functionDef.parameters();
    if (parametersList != null) {
      createParameterNames(parametersList.all(), fileId);
    }
    functionDefinitionLocation = locationInFile(functionDef.name(), fileId);
  }

  FunctionSymbolImpl(String name, FunctionSymbol functionSymbol) {
    super(name, functionSymbol.fullyQualifiedName());
    setKind(Kind.FUNCTION);
    isInstanceMethod = functionSymbol.isInstanceMethod();
    hasDecorators = functionSymbol.hasDecorators();
    hasVariadicParameter = functionSymbol.hasVariadicParameter();
    parameters.addAll(functionSymbol.parameters());
    functionDefinitionLocation = functionSymbol.definitionLocation();
    declaredReturnType = ((FunctionSymbolImpl) functionSymbol).declaredReturnType();
    isStub = functionSymbol.isStub();
  }

  public FunctionSymbolImpl(String name, @Nullable String fullyQualifiedName, boolean hasVariadicParameter,
                     boolean isInstanceMethod, boolean hasDecorators, List<Parameter> parameters) {
    super(name, fullyQualifiedName);
    setKind(Kind.FUNCTION);
    this.hasVariadicParameter = hasVariadicParameter;
    this.isInstanceMethod = isInstanceMethod;
    this.hasDecorators = hasDecorators;
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
      .noneMatch(decorator -> decorator.equals("staticmethod") || decorator.equals("classmethod"));
  }

  private void createParameterNames(List<AnyParameter> parameterTrees, @Nullable String fileId) {
    ParameterState parameterState = new ParameterState();
    for (AnyParameter anyParameter : parameterTrees) {
      if (anyParameter.is(Tree.Kind.PARAMETER)) {
        addParameter((org.sonar.plugins.python.api.tree.Parameter) anyParameter, fileId, parameterState);
      } else {
        parameters.add(new ParameterImpl(null, false, parameterState, locationInFile(anyParameter, fileId)));
      }
    }
  }

  private void addParameter(org.sonar.plugins.python.api.tree.Parameter parameter, @Nullable String fileId, ParameterState parameterState) {
    Name parameterName = parameter.name();
    Token starToken = parameter.starToken();
    if (parameterName != null) {
      this.parameters.add(new ParameterImpl(parameterName.name(), parameter.defaultValue() != null, parameterState, locationInFile(parameter, fileId)));
      if (starToken != null) {
        hasVariadicParameter = true;
      }
    } else if (starToken != null) {
      if ("*".equals(starToken.value())) {
        parameterState.keywordOnly = true;
        parameterState.positionalOnly = false;
      }
      if ("/".equals(starToken.value())) {
        parameterState.positionalOnly = true;
      }
    }
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
    private final boolean hasDefaultValue;
    private final boolean isKeywordOnly;
    private final boolean isPositionalOnly;
    private final LocationInFile location;

    ParameterImpl(@Nullable String name, boolean hasDefaultValue, ParameterState parameterState, @Nullable LocationInFile location) {
      this.name = name;
      this.hasDefaultValue = hasDefaultValue;
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
    public boolean hasDefaultValue() {
      return hasDefaultValue;
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
