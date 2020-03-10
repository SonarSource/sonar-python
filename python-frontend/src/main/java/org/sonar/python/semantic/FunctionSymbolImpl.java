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
    ParameterList parametersList = functionDef.parameters();
    if (parametersList != null) {
      createParameterNames(parametersList.all());
    }
    if (SymbolUtils.isTypeShedFile(pythonFile)) {
      functionDefinitionLocation = null;
    } else {
      TokenLocation functionName = new TokenLocation(functionDef.name().firstToken());
      Path path = pathOf(pythonFile);
      String fileId = path != null ? path.toString() : pythonFile.toString();
      functionDefinitionLocation = new LocationInFile(fileId, functionName.startLine(), functionName.startLineOffset(), functionName.endLine(), functionName.endLineOffset());
    }
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

  @Override
  FunctionSymbolImpl copyWithoutUsages() {
    return new FunctionSymbolImpl(name(), this);
  }

  private static boolean isInstanceMethod(FunctionDef functionDef) {
    return functionDef.isMethodDefinition() && functionDef.decorators().stream()
      .map(decorator -> {
        List<Name> names = decorator.name().names();
        return names.get(names.size() - 1).name();
      })
      .noneMatch(decorator -> decorator.equals("staticmethod") || decorator.equals("classmethod"));
  }

  private void createParameterNames(List<AnyParameter> parameterTrees) {
    boolean keywordOnly = false;
    for (AnyParameter anyParameter : parameterTrees) {
      if (anyParameter.is(Tree.Kind.PARAMETER)) {
        org.sonar.plugins.python.api.tree.Parameter parameter = (org.sonar.plugins.python.api.tree.Parameter) anyParameter;
        Name parameterName = parameter.name();
        this.parameters.add(new ParameterImpl(parameterName.name(), parameter.defaultValue() != null, keywordOnly));
        if (parameter.starToken() != null) {
          hasVariadicParameter = true;
        }
      } else if (anyParameter.is(Tree.Kind.SEPARATOR_PARAMETER)) {
        if (anyParameter.firstToken().value().equals("*")) {
          keywordOnly = true;
        }
      } else {
        parameters.add(new ParameterImpl(null, false, false));
      }
    }
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

    ParameterImpl(@Nullable String name, boolean hasDefaultValue, boolean isKeywordOnly) {
      this.name = name;
      this.hasDefaultValue = hasDefaultValue;
      this.isKeywordOnly = isKeywordOnly;
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
  }
}
