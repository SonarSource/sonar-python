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
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;

/**
 * ClassType
 */
public record ClassType(
  String name,
  Set<Member> members,
  List<PythonType> attributes,
  List<PythonType> superClasses,
  List<PythonType> metaClasses,
  @Nullable LocationInFile locationInFile) implements PythonType {

  public ClassType(String name) {
    this(name, new HashSet<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), null);
  }

  public ClassType(String name, @Nullable LocationInFile locationInFile) {
    this(name, new HashSet<>(), new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), locationInFile);
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

  public boolean isASubClassFrom(ClassType other) {
    return superClasses.stream().anyMatch(superClass -> superClass.isCompatibleWith(other));
  }

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
    return superClasses.stream()
      .map(s -> s.resolveMember(memberName))
      .filter(Optional::isPresent)
      .map(Optional::get)
      .findFirst();
  }

  public boolean hasUnresolvedHierarchy() {
    return superClasses.stream().anyMatch(s -> {
        if (s instanceof ClassType parentClassType) {
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
    return !this.metaClasses.isEmpty();
  }

  public TriBool instancesHaveMember(String memberName) {
    if (hasUnresolvedHierarchy() || hasMetaClass()) {
      return TriBool.UNKNOWN;
    }
    if ("NamedTuple".equals(this.name)) {
      // TODO: instances of NamedTuple are type
      return TriBool.TRUE;
    }
    return resolveMember(memberName).isPresent() ? TriBool.TRUE : TriBool.FALSE;
  }

  @Override
  public Optional<LocationInFile> definitionLocation() {
    return Optional.ofNullable(this.locationInFile);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    ClassType classType = (ClassType) o;
    boolean haveSameAttributes = Objects.equals(name, classType.name) && Objects.equals(members, classType.members) && Objects.equals(attributes, classType.attributes);
    List<String> parentNames = superClasses.stream().map(PythonType::key).toList();
    List<String> metaClassNames = metaClasses.stream().map(PythonType::key).toList();
    List<String> otherParentNames = classType.superClasses.stream().map(PythonType::key).toList();
    List<String> otherMetaClassNames = classType.metaClasses.stream().map(PythonType::key).toList();
    return haveSameAttributes && Objects.equals(parentNames, otherParentNames) && Objects.equals(metaClassNames, otherMetaClassNames);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, members, attributes, superClasses);
  }

  @Override
  public String toString() {
    return "ClassType[%s]".formatted(name);
  }
}
