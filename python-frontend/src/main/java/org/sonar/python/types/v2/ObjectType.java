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
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.LocationInFile;

@Beta
public final class ObjectType implements PythonType {
  private final TypeWrapper typeWrapper;
  private final List<PythonType> attributes;
  private final List<Member> members;
  private final TypeSource typeSource;

  public ObjectType(TypeWrapper typeWrapper, List<PythonType> attributes, List<Member> members, TypeSource typeSource) {
    this.typeWrapper = typeWrapper;
    this.attributes = attributes;
    this.members = members;
    this.typeSource = typeSource;
  }

  public ObjectType(PythonType type, List<PythonType> attributes, List<Member> members, TypeSource typeSource) {
    this(TypeWrapper.of(type), attributes, members, typeSource);
  }

  public ObjectType(PythonType type) {
    this(type, TypeSource.EXACT);
  }

  public ObjectType(TypeWrapper typeWrapper) {
    this(typeWrapper, new ArrayList<>(), new ArrayList<>(), TypeSource.EXACT);
  }

  public ObjectType(PythonType type, TypeSource typeSource) {
    this(type, new ArrayList<>(), new ArrayList<>(), typeSource);
  }

  public ObjectType(PythonType type, List<PythonType> attributes, List<Member> members) {
    this(type, attributes, members, TypeSource.EXACT);
  }

  @Override
  public Optional<String> displayName() {
    return typeWrapper.type().instanceDisplayName();
  }

  @Override
  public boolean isCompatibleWith(PythonType another) {
    return this.typeWrapper.type().isCompatibleWith(another);
  }

  @Override
  public PythonType unwrappedType() {
    return this.typeWrapper.type();
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return members().stream()
      .filter(member -> Objects.equals(member.name(), memberName))
      .map(Member::type)
      .findFirst()
      .or(() -> typeWrapper.type().resolveMember(memberName));
  }

  @Override
  public TriBool hasMember(String memberName) {
    if (resolveMember(memberName).isPresent()) {
      return TriBool.TRUE;
    }
    if (typeWrapper.type() instanceof ClassType classType) {
      return classType.instancesHaveMember(memberName);
    }
    return TriBool.UNKNOWN;
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return typeWrapper.type().definitionLocation();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    ObjectType that = (ObjectType) o;
    List<String> membersNames = members.stream().map(Member::name).toList();
    List<String> otherMembersNames = that.members.stream().map(Member::name).toList();
    List<String> attributesNames = attributes.stream().map(PythonType::key).toList();
    List<String> otherAttributesNames = that.attributes.stream().map(PythonType::key).toList();
    return Objects.equals(typeWrapper, that.typeWrapper) && Objects.equals(membersNames, otherMembersNames) && Objects.equals(attributesNames, otherAttributesNames);
  }

  @Override
  public int hashCode() {
    List<String> membersNames = members.stream().map(Member::name).toList();
    List<String> attributesNames = attributes.stream().map(PythonType::key).toList();
    return Objects.hash(typeWrapper, attributesNames, membersNames);
  }

  public PythonType type() {
    return typeWrapper.type();
  }

  public TypeWrapper typeWrapper() {
    return typeWrapper;
  }

  public List<PythonType> attributes() {
    return attributes;
  }

  public List<Member> members() {
    return members;
  }

  @Override
  public TypeSource typeSource() {
    return typeSource;
  }

  @Override
  public String toString() {
    return "ObjectType[" +
      "type=" + typeWrapper + ", " +
      "attributes=" + attributes + ", " +
      "members=" + members + ", " +
      "typeSource=" + typeSource + ']';
  }

}
