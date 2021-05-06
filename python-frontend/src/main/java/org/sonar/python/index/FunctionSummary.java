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
package org.sonar.python.index;

import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;

public class FunctionSummary implements Summary {

  private final String name;
  private final String fullyQualifiedName;
  private final List<Parameter> parameters;
  private final boolean isAsynchronous;
  private final boolean isInstanceMethod;
  private final List<String> decorators;
  private final boolean hasDecorators;
  private final LocationInFile definitionLocation;
  private final String annotatedReturnTypeName;

  private FunctionSummary(String name, String fullyQualifiedName, List<Parameter> parameters, boolean isAsynchronous,
    boolean isInstanceMethod, List<String> decorators, boolean hasDecorators, LocationInFile definitionLocation, String annotatedReturnTypeName) {

    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.parameters = parameters;
    this.isAsynchronous = isAsynchronous;
    this.isInstanceMethod = isInstanceMethod;
    this.decorators = decorators;
    this.hasDecorators = hasDecorators;
    this.definitionLocation = definitionLocation;
    this.annotatedReturnTypeName = annotatedReturnTypeName;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  public List<Parameter> parameters() {
    return parameters;
  }

  public boolean isAsynchronous() {
    return isAsynchronous;
  }

  public boolean isInstanceMethod() {
    return isInstanceMethod;
  }

  public List<String> decorators() {
    return decorators;
  }

  public boolean hasDecorators() {
    return hasDecorators;
  }

  @CheckForNull
  public LocationInFile definitionLocation() {
    return definitionLocation;
  }

  @CheckForNull
  public String annotatedReturnTypeName() {
    return annotatedReturnTypeName;
  }

  public static class Parameter  {

    private final String name;
    private final String annotatedType;
    private final boolean hasDefaultValue;
    private final boolean isVariadic;
    private final boolean isKeywordOnly;
    private final boolean isPositionalOnly;
    private final LocationInFile location;

    public Parameter(@Nullable String name, String annotatedType, boolean hasDefaultValue,
      boolean isVariadic, boolean isKeywordOnly, boolean isPositionalOnly, @Nullable LocationInFile location) {
      this.name = name;
      this.annotatedType = annotatedType;
      this.hasDefaultValue = hasDefaultValue;
      this.isVariadic = isVariadic;
      this.isKeywordOnly = isKeywordOnly;
      this.isPositionalOnly = isPositionalOnly;
      this.location = location;
    }

    @CheckForNull
    public String name() {
      return name;
    }

    public String annotatedType() {
      return annotatedType;
    }

    public boolean hasDefaultValue() {
      return hasDefaultValue;
    }

    public boolean isVariadic() {
      return isVariadic;
    }

    public boolean isKeywordOnly() {
      return isKeywordOnly;
    }

    public boolean isPositionalOnly() {
      return isPositionalOnly;
    }

    @CheckForNull
    public LocationInFile location() {
      return location;
    }
  }

  public static class FunctionSummaryBuilder {

    private String name;
    private String fullyQualifiedName;
    private List<Parameter> parameters;
    private boolean isAsynchronous;
    private boolean isInstanceMethod;
    private List<String> decorators;
    private boolean hasDecorators;
    private LocationInFile definitionLocation;
    private String annotatedReturnTypeName;

    public FunctionSummaryBuilder withName(String name) {
      this.name = name;
      return this;
    }

    public FunctionSummaryBuilder withFullyQualifiedName(String fullyQualifiedName) {
      this.fullyQualifiedName = fullyQualifiedName;
      return this;
    }

    public FunctionSummaryBuilder withParameters(List<Parameter> parameters) {
      this.parameters = parameters;
      return this;
    }

    public FunctionSummaryBuilder withIsAsynchronous(boolean isAsynchronous) {
      this.isAsynchronous = isAsynchronous;
      return this;
    }

    public FunctionSummaryBuilder withIsInstanceMethod(boolean isInstanceMethod) {
      this.isInstanceMethod = isInstanceMethod;
      return this;
    }

    public FunctionSummaryBuilder withDecorators(List<String> decorators) {
      this.decorators = decorators;
      return this;
    }

    public FunctionSummaryBuilder withHasDecorators(boolean hasDecorators) {
      this.hasDecorators = hasDecorators;
      return this;
    }

    public FunctionSummaryBuilder withDefinitionLocation(LocationInFile definitionLocation) {
      this.definitionLocation = definitionLocation;
      return this;
    }

    public FunctionSummaryBuilder withAnnotatedReturnTypeName(String annotatedReturnTypeName) {
      this.annotatedReturnTypeName = annotatedReturnTypeName;
      return this;
    }

    public FunctionSummary build() {
      return new FunctionSummary(name, fullyQualifiedName, parameters, isAsynchronous, isInstanceMethod, decorators,
        hasDecorators, definitionLocation, annotatedReturnTypeName);
    }
  }
}
