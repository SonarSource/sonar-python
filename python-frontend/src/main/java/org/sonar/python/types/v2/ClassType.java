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
package org.sonar.python.types.v2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.LocationInFile;

/**
 * ClassType
 */
@Beta
public final class ClassType implements PythonType {

  private final String name;
  private final String fullyQualifiedName;
  private final Set<Member> members;
  private final List<PythonType> attributes;
  private final List<TypeWrapper> superClasses;
  private final List<PythonType> metaClasses;
  private final boolean hasDecorators;
  private final boolean isGeneric;
  private final LocationInFile locationInFile;

  public ClassType(
    String name,
    String fullyQualifiedName,
    Set<Member> members,
    List<PythonType> attributes,
    List<TypeWrapper> superClasses,
    List<PythonType> metaClasses,
    boolean hasDecorators,
    boolean isGeneric,
    @Nullable LocationInFile locationInFile) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.members = members;
    this.attributes = attributes;
    this.superClasses = superClasses;
    this.metaClasses = metaClasses;
    this.hasDecorators = hasDecorators;
    this.isGeneric = isGeneric;
    this.locationInFile = locationInFile;
  }

  public ClassType(String name, String fullyQualifiedName) {
    this(name, fullyQualifiedName, new HashSet<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), false, false, null);
  }

  @Override
  public Optional<String> displayName() {
    return Optional.of("type");
  }

  @Override
  public Optional<String> instanceDisplayName() {
    var splits = name.split("\\.");
    if (splits.length > 0) {
      return Optional.of(splits[splits.length - 1]);
    }
    return Optional.of(name);
  }

  @Override
  public boolean isCompatibleWith(PythonType another) {
    if (another instanceof ObjectType objectType) {
      return this.isCompatibleWith(objectType.type());
    }
    if (another instanceof UnionType unionType) {
      return unionType.candidates().stream().anyMatch(this::isCompatibleWith);
    }
    if (another instanceof FunctionType functionType) {
      return this.isCompatibleWith(functionType.returnType());
    }
    if (another instanceof ClassType classType) {
      var isASubClass = this.isASubClassFrom(classType);
      var areAttributeCompatible = this.areAttributesCompatible(classType);
      var isDuckTypeCompatible = !this.members.isEmpty() && this.members.containsAll(classType.members);
      return Objects.equals(this, another)
        || "builtins.object".equals(classType.name())
        || isDuckTypeCompatible
        || (isASubClass && areAttributeCompatible);
    }
    return true;
  }

  @Beta
  public boolean isASubClassFrom(ClassType other) {
    return superClasses().stream().anyMatch(superClass -> superClass.type().isCompatibleWith(other));
  }

  @Beta
  public boolean areAttributesCompatible(ClassType other) {
    return attributes.stream().allMatch(attr -> other.attributes.stream().anyMatch(attr::isCompatibleWith));
  }

  @Override
  public String key() {
    return Optional.of(attributes())
      .stream()
      .flatMap(Collection::stream)
      .map(PythonType::key)
      .collect(Collectors.joining(",", name() + "[", "]"));
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return localMember(memberName)
      .or(() -> inheritedMember(memberName));
  }

  private Optional<PythonType> localMember(String memberName) {
    return members.stream()
      .filter(m -> m.name().equals(memberName))
      .map(Member::type)
      .findFirst();
  }

  private Optional<PythonType> inheritedMember(String memberName) {
    return superClasses().stream()
      .map(s -> s.type().resolveMember(memberName))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .findFirst();
  }

  public boolean hasUnresolvedHierarchy() {
    return superClasses().stream().anyMatch(s -> {
        if (s.type() instanceof ClassType parentClassType) {
          return parentClassType.hasUnresolvedHierarchy();
        }
        return true;
      }
    );
  }

  @Override
  public TriBool hasMember(String memberName) {
    // a ClassType is an object of class type, it has the same members as those present on any type
    if ("__call__".equals(memberName)) {
      return TriBool.TRUE;
    }
    if (hasUnresolvedHierarchy()) {
      return TriBool.UNKNOWN;
    }
    // TODO: Not correct, we should look at what the actual type is instead (SONARPY-1666)
    return TriBool.UNKNOWN;
  }

  public boolean hasMetaClass() {
    return !this.metaClasses.isEmpty() ||
      this.superClasses()
        .stream()
        .map(TypeWrapper::type)
        .filter(ClassType.class::isInstance)
        .map(ClassType.class::cast)
        .anyMatch(ClassType::hasMetaClass);
  }

  public TriBool instancesHaveMember(String memberName) {
    if (hasUnresolvedHierarchy() || hasMetaClass() || hasDecorators()) {
      return TriBool.UNKNOWN;
    }
    if ("NamedTuple".equals(this.name)) {
      // TODO: instances of NamedTuple are type
      return TriBool.TRUE;
    }
    if ("function".equals(this.name) && "__call__".equals(memberName)) {
      // __call__ is not formally defined on function, but is present
      return TriBool.TRUE;
    }
    return resolveMember(memberName).isPresent() ? TriBool.TRUE : TriBool.FALSE;
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return Optional.ofNullable(this.locationInFile);
  }

  @Override
  public String toString() {
    return "ClassType[%s]".formatted(name);
  }

  @Override
  public String name() {
    return name;
  }

  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  public Set<Member> members() {
    return members;
  }

  public List<PythonType> attributes() {
    return attributes;
  }

  public List<TypeWrapper> superClasses() {
    return superClasses;
  }

  public List<PythonType> metaClasses() {
    return metaClasses;
  }

  public boolean hasDecorators() {
    return hasDecorators;
  }

  public boolean isGeneric() {
    return isGeneric;
  }
}
