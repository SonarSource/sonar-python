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
package org.sonar.python.types.v2;

import java.util.List;

/**
 * FunctionType
 */
public class FunctionType implements PythonType {
  private boolean hasVariadicParameter;
  private final String name;
  private final List<PythonType> attributes;
  private final List<ParameterV2> parameters;
  private boolean isAsynchronous;
  private boolean hasDecorators;
  private boolean isInstanceMethod;
  private PythonType owner;
  private PythonType returnType = PythonType.UNKNOWN;


  public FunctionType(String name, List<PythonType> attributes, List<ParameterV2> parameters, PythonType returnType) {
    this.name = name;
    this.attributes = attributes;
    this.parameters = parameters;
    this.returnType = returnType;
  }

  public FunctionType(String name, List<PythonType> attributes, List<ParameterV2> parameters, PythonType returnType,
    boolean isAsynchronous, boolean hasDecorators, boolean isInstanceMethod, boolean hasVariadicParameter, PythonType owner) {
    this.name = name;
    this.attributes = attributes;
    this.parameters = parameters;
    this.returnType = returnType;
    this.isAsynchronous = isAsynchronous;
    this.hasDecorators = hasDecorators;
    this.isInstanceMethod = isInstanceMethod;
    this.hasVariadicParameter = hasVariadicParameter;
    this.owner = owner;
  }

  public static class ParameterState {
    public boolean keywordOnly = false;
    public boolean positionalOnly = false;
  }

  public record ParameterType(PythonType pythonType, boolean isKeywordVariadic, boolean isPositionalVariadic) { }

  public boolean hasVariadicParameter() {
    return hasVariadicParameter;
  }

  public PythonType returnType() {
    return returnType;
  }

  public String name() {
    return name;
  }

  public List<PythonType> attributes() {
    return attributes;
  }

  public List<ParameterV2> parameters() {
    return parameters;
  }

  public boolean isAsynchronous() {
    return isAsynchronous;
  }

  public boolean hasDecorators() {
    return hasDecorators;
  }

  public boolean isInstanceMethod() {
    return isInstanceMethod;
  }

  public PythonType owner() {
    return owner;
  }
}
