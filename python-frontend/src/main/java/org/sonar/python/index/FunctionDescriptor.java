/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.types.protobuf.DescriptorsProtos;

public class FunctionDescriptor implements Descriptor {

  private final String name;
  @Nullable
  private final String fullyQualifiedName;
  private final List<Parameter> parameters;
  private final boolean isAsynchronous;
  private final boolean isInstanceMethod;
  private final List<String> decorators;
  private final boolean hasDecorators;
  @Nullable
  private final LocationInFile definitionLocation;
  @Nullable
  private final String annotatedReturnTypeName;

  private FunctionDescriptor(String name, @Nullable String fullyQualifiedName, List<Parameter> parameters, boolean isAsynchronous,
    boolean isInstanceMethod, List<String> decorators, boolean hasDecorators, @Nullable LocationInFile definitionLocation, @Nullable String annotatedReturnTypeName) {

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

  public FunctionDescriptor(DescriptorsProtos.FunctionDescriptor functionDescriptorProto) {
    this.name = functionDescriptorProto.getName();
    this.fullyQualifiedName = functionDescriptorProto.getFullyQualifiedName();
    this.parameters = new ArrayList<>();
    functionDescriptorProto.getParametersList().forEach(proto -> parameters.add(new Parameter(proto)));
    this.isAsynchronous = functionDescriptorProto.getIsAsynchronous();
    this.isInstanceMethod = functionDescriptorProto.getIsInstanceMethod();
    this.decorators = new ArrayList<>(functionDescriptorProto.getDecoratorsList());
    this.hasDecorators = functionDescriptorProto.getHasDecorators();
    this.definitionLocation = new LocationInFile(functionDescriptorProto.getDefinitionLocation());
    this.annotatedReturnTypeName = functionDescriptorProto.getAnnotatedReturnType();
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  @Override
  public Kind kind() {
    return Kind.FUNCTION;
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
    private final boolean isKeywordVariadic;
    private final boolean isPositionalVariadic;
    private final boolean isKeywordOnly;
    private final boolean isPositionalOnly;
    private final LocationInFile location;

    public Parameter(@Nullable String name, @Nullable String annotatedType, boolean hasDefaultValue,
                     boolean isKeywordOnly, boolean isPositionalOnly, boolean isPositionalVariadic, boolean isKeywordVariadic, @Nullable LocationInFile location) {
      this.name = name;
      this.annotatedType = annotatedType;
      this.hasDefaultValue = hasDefaultValue;
      this.isKeywordVariadic = isKeywordVariadic;
      this.isPositionalVariadic = isPositionalVariadic;
      this.isKeywordOnly = isKeywordOnly;
      this.isPositionalOnly = isPositionalOnly;
      this.location = location;
    }

    public Parameter(DescriptorsProtos.ParameterDescriptor parameterDescriptorProto) {
      this.name = parameterDescriptorProto.getName();
      this.annotatedType = parameterDescriptorProto.getAnnotatedType();
      this.hasDefaultValue = parameterDescriptorProto.getHasDefaultValue();
      this.isKeywordVariadic = parameterDescriptorProto.getIsKeywordVariadic();
      this.isPositionalVariadic = parameterDescriptorProto.getIsPositionalVariadic();
      this.isKeywordOnly = parameterDescriptorProto.getIsKeywordOnly();
      this.isPositionalOnly = parameterDescriptorProto.getIsPositionalOnly();
      this.location = new LocationInFile(parameterDescriptorProto.getDefinitionLocation());
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
      return isKeywordVariadic || isPositionalVariadic;
    }

    public boolean isKeywordOnly() {
      return isKeywordOnly;
    }

    public boolean isPositionalOnly() {
      return isPositionalOnly;
    }

    public boolean isKeywordVariadic() {
      return isKeywordVariadic;
    }

    public boolean isPositionalVariadic() {
      return isPositionalVariadic;
    }

    @CheckForNull
    public LocationInFile location() {
      return location;
    }

    public DescriptorsProtos.ParameterDescriptor toProtobuf() {
      DescriptorsProtos.ParameterDescriptor.Builder builder = DescriptorsProtos.ParameterDescriptor.newBuilder()
        .setName(name)
        .setAnnotatedType(annotatedType)
        .setHasDefaultValue(hasDefaultValue)
        .setIsKeywordVariadic(isKeywordVariadic)
        .setIsPositionalVariadic(isPositionalVariadic)
        .setIsKeywordOnly(isKeywordOnly)
        .setIsPositionalOnly(isPositionalOnly);
      if (location != null) {
        builder.setDefinitionLocation(location.toProtobuf());
      }
      return builder.build();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      Parameter parameter = (Parameter) o;
      return hasDefaultValue == parameter.hasDefaultValue &&
        isKeywordVariadic == parameter.isKeywordVariadic &&
        isPositionalVariadic == parameter.isPositionalVariadic &&
        isKeywordOnly == parameter.isKeywordOnly &&
        isPositionalOnly == parameter.isPositionalOnly &&
        Objects.equals(name, parameter.name) &&
        Objects.equals(annotatedType, parameter.annotatedType) &&
        Objects.equals(location, parameter.location);
    }

    @Override
    public int hashCode() {
      return Objects.hash(name, annotatedType, hasDefaultValue, isKeywordVariadic, isPositionalVariadic, isKeywordOnly, isPositionalOnly, location);
    }
  }

  public static class FunctionDescriptorBuilder {

    private String name;
    private String fullyQualifiedName;
    private List<Parameter> parameters = new ArrayList<>();
    private boolean isAsynchronous = false;
    private boolean isInstanceMethod = false;
    private List<String> decorators = new ArrayList<>();
    private boolean hasDecorators = false;
    private LocationInFile definitionLocation = null;
    private String annotatedReturnTypeName = null;

    public FunctionDescriptorBuilder withName(String name) {
      this.name = name;
      return this;
    }

    public FunctionDescriptorBuilder withFullyQualifiedName(@Nullable String fullyQualifiedName) {
      this.fullyQualifiedName = fullyQualifiedName;
      return this;
    }

    public FunctionDescriptorBuilder withParameters(List<Parameter> parameters) {
      this.parameters = parameters;
      return this;
    }

    public FunctionDescriptorBuilder withIsAsynchronous(boolean isAsynchronous) {
      this.isAsynchronous = isAsynchronous;
      return this;
    }

    public FunctionDescriptorBuilder withIsInstanceMethod(boolean isInstanceMethod) {
      this.isInstanceMethod = isInstanceMethod;
      return this;
    }

    public FunctionDescriptorBuilder withDecorators(List<String> decorators) {
      this.decorators = decorators;
      return this;
    }

    public FunctionDescriptorBuilder withHasDecorators(boolean hasDecorators) {
      this.hasDecorators = hasDecorators;
      return this;
    }

    public FunctionDescriptorBuilder withDefinitionLocation(@Nullable LocationInFile definitionLocation) {
      this.definitionLocation = definitionLocation;
      return this;
    }

    public FunctionDescriptorBuilder withAnnotatedReturnTypeName(@Nullable String annotatedReturnTypeName) {
      this.annotatedReturnTypeName = annotatedReturnTypeName;
      return this;
    }

    public FunctionDescriptor build() {
      return new FunctionDescriptor(name, fullyQualifiedName, parameters, isAsynchronous, isInstanceMethod, decorators,
        hasDecorators, definitionLocation, annotatedReturnTypeName);
    }
  }

  public DescriptorsProtos.FunctionDescriptor toProtobuf() {
    DescriptorsProtos.FunctionDescriptor.Builder builder = DescriptorsProtos.FunctionDescriptor.newBuilder()
      .setName(name)
      .setFullyQualifiedName(fullyQualifiedName)
      .addAllParameters(parameters.stream().map(Parameter::toProtobuf).collect(Collectors.toList()))
      .setIsAsynchronous(isAsynchronous)
      .setIsInstanceMethod(isInstanceMethod)
      .addAllDecorators(decorators)
      .setHasDecorators(hasDecorators)
      .setAnnotatedReturnType(annotatedReturnTypeName);
    if (definitionLocation != null) {
      builder.setDefinitionLocation(definitionLocation.toProtobuf());
    }
    return builder.build();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    FunctionDescriptor that = (FunctionDescriptor) o;
    return isAsynchronous == that.isAsynchronous &&
      isInstanceMethod == that.isInstanceMethod &&
      hasDecorators == that.hasDecorators && name.equals(that.name) &&
      Objects.equals(fullyQualifiedName, that.fullyQualifiedName) &&
      parameters.equals(that.parameters) && decorators.equals(that.decorators) &&
      Objects.equals(definitionLocation, that.definitionLocation) &&
      Objects.equals(annotatedReturnTypeName, that.annotatedReturnTypeName);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, fullyQualifiedName, parameters, isAsynchronous, isInstanceMethod, decorators, hasDecorators, definitionLocation, annotatedReturnTypeName);
  }
}
