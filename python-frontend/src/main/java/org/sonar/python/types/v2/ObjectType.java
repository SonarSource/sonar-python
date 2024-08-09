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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.LocationInFile;

@Beta
public record ObjectType(PythonType type, List<PythonType> attributes, List<Member> members, TypeSource typeSource) implements PythonType {

  public ObjectType(PythonType type) {
    this(type, TypeSource.EXACT);
  }

  public ObjectType(PythonType type, TypeSource typeSource) {
    this(type, new ArrayList<>(), new ArrayList<>(), typeSource);
  }

  public ObjectType(PythonType type, List<PythonType> attributes, List<Member> members) {
    this(type, attributes, members, TypeSource.EXACT);
  }

  @Override
  public Optional<String> displayName() {
    return type.instanceDisplayName();
  }

  @Override
  public boolean isCompatibleWith(PythonType another) {
    return this.type.isCompatibleWith(another);
  }

  @Override
  public PythonType unwrappedType() {
    return this.type;
  }

  @Override
  public Optional<PythonType> resolveMember(String memberName) {
    return members().stream()
      .filter(member -> Objects.equals(member.name(), memberName))
      .map(Member::type)
      .findFirst()
      .or(() -> type.resolveMember(memberName));
  }

  @Override
  public TriBool hasMember(String memberName) {
    if (resolveMember(memberName).isPresent()) {
      return TriBool.TRUE;
    }
    if (type instanceof ClassType classType) {
      return classType.instancesHaveMember(memberName);
    }
    return TriBool.UNKNOWN;
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return type.definitionLocation();
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
    return Objects.equals(type, that.type) && Objects.equals(membersNames, otherMembersNames) && Objects.equals(attributesNames, otherAttributesNames);
  }

  @Override
  public int hashCode() {
    List<String> membersNames = members.stream().map(Member::name).toList();
    List<String> attributesNames = attributes.stream().map(PythonType::key).toList();
    return Objects.hash(type, attributesNames, membersNames);
  }
}
