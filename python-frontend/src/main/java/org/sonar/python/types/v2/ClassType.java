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
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * ClassType
 */
public record ClassType(
  String name,
  Set<Member> members,
  List<PythonType> attributes,
  List<PythonType> superClasses) implements PythonType {

  public ClassType(String name) {
    this(name, new HashSet<>(), new ArrayList<>(), new ArrayList<>());
  }

  public ClassType(String name, List<PythonType> attributes) {
    this(name, new HashSet<>(), attributes, new ArrayList<>());
  }

  @Override
  public String displayName() {
    var splits = name.split("\\.");
    if (splits.length > 0) {
      return splits[splits.length - 1];
    }
    return name;
  }

  @Override
  public boolean isCompatibleWith(PythonType another) {
    if (another instanceof ObjectType) {
      return this.isCompatibleWith(((ObjectType) another).type());
    }
    if (another instanceof UnionType) {
      return ((UnionType) another).candidates().stream().anyMatch(c -> this.isCompatibleWith(c));
    }
    if (another instanceof FunctionType) {
      return this.isCompatibleWith(((FunctionType) another).returnType());
    }
    if (another instanceof ClassType) {
      var other = (ClassType) another;
      var isASubClass = this.isASubClassFrom(other);
      var areAttributeCompatible = this.areAttributesCompatible(other);
      var isDuckTypeCompatible = !this.members.isEmpty() && other.members.stream().allMatch(member -> this.members.contains(member));
      return Objects.equals(this, another) || "builtins.object".equals(other.name()) || 
        isDuckTypeCompatible ||
        ( isASubClass && areAttributeCompatible) ;
    }
    return true;
  }

  public boolean isASubClassFrom(ClassType other) {
    return superClasses.stream().anyMatch(superClass -> superClass.isCompatibleWith(other));
  }

  public boolean areAttributesCompatible(ClassType other) {
    return attributes.stream().allMatch(attr -> other.attributes.stream().anyMatch(otherAttr -> attr.isCompatibleWith(otherAttr)));
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
  public PythonType resolveMember(String memberName) {
    return members.stream()
      .filter(m -> m.name().equals(memberName))
      .map(Member::type)
      .findFirst().orElse(PythonType.UNKNOWN);
  }
}
