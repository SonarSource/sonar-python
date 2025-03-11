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
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.Member;
import org.sonar.plugins.python.api.types.v2.PythonType;

public class ClassTypeBuilder implements TypeBuilder<ClassType> {

  private final String name;
  private final String fullyQualifiedName;
  Set<Member> members = new HashSet<>();
  List<PythonType> attributes = new ArrayList<>();
  List<TypeWrapper> superClasses = new ArrayList<>();
  List<PythonType> metaClasses = new ArrayList<>();
  boolean hasDecorators = false;
  boolean isGeneric = false;
  LocationInFile definitionLocation;

  @Override
  public ClassType build() {
    return new ClassType(name, fullyQualifiedName, members, attributes, superClasses, metaClasses, hasDecorators, isGeneric, definitionLocation);
  }

  public ClassTypeBuilder(String name, String fullyQualifiedName) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
  }

  public ClassTypeBuilder withHasDecorators(boolean hasDecorators) {
    this.hasDecorators = hasDecorators;
    return this;
  }

  public ClassTypeBuilder withIsGeneric(boolean isGeneric) {
    this.isGeneric = isGeneric;
    return this;
  }

  @Override
  public ClassTypeBuilder withDefinitionLocation(@Nullable LocationInFile definitionLocation) {
    this.definitionLocation = definitionLocation;
    return this;
  }

  public ClassTypeBuilder addSuperClass(PythonType type) {
    superClasses.add(TypeWrapper.of(type));
    return this;
  }

  public ClassTypeBuilder addSuperClass(TypeWrapper typeWrapper) {
    superClasses.add(typeWrapper);
    return this;
  }

  public ClassTypeBuilder withSuperClasses(PythonType... types) {
    Arrays.stream(types).forEach(this::addSuperClass);
    return this;
  }

  public List<PythonType> metaClasses() {
    return metaClasses;
  }
}
