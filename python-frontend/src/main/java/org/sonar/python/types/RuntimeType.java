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
package org.sonar.python.types;

import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;

class RuntimeType implements InferredType {

  private final ClassSymbol typeClass;
  private Set<String> typeClassSuperClassesFQN = null;
  private Set<String> typeClassMembersFQN = null;

  RuntimeType(ClassSymbol typeClass) {
    this.typeClass = typeClass;
  }

  @Override
  public boolean isIdentityComparableWith(InferredType other) {
    if (other == AnyType.ANY || other instanceof DeclaredType) {
      return true;
    }
    if (other instanceof UnionType) {
      return other.isIdentityComparableWith(this);
    }
    return this.equals(other);
  }

  @Override
  public boolean canHaveMember(String memberName) {
    return typeClass.canHaveMember(memberName);
  }

  @Override
  public boolean declaresMember(String memberName) {
    return canHaveMember(memberName);
  }

  @Override
  public Optional<Symbol> resolveMember(String memberName) {
    return typeClass.resolveMember(memberName);
  }

  @Override
  public Optional<Symbol> resolveDeclaredMember(String memberName) {
    return resolveMember(memberName);
  }

  @Override
  public boolean canOnlyBe(String typeName) {
    return typeName.equals(typeClass.fullyQualifiedName());
  }

  @Override
  public boolean canBeOrExtend(String typeName) {
    return typeClass.canBeOrExtend(typeName);
  }

  @Override
  public boolean isCompatibleWith(InferredType other) {
    return InferredTypes.isTypeClassCompatibleWith(typeClass, other);
  }

  public boolean mustBeOrExtend(String fullyQualifiedName) {
    return typeClass.isOrExtends(fullyQualifiedName);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    RuntimeType that = (RuntimeType) o;
    return Objects.equals(typeClass.name(), that.typeClass.name()) &&
      Objects.equals(typeClass.fullyQualifiedName(), that.typeClass.fullyQualifiedName())
      && Objects.equals(typeClassSuperClassesFQN(), that.typeClassSuperClassesFQN())
      && Objects.equals(typeClassMembersFQN(), that.typeClassMembersFQN());
  }

  private Set<String> typeClassSuperClassesFQN() {
    if (typeClassSuperClassesFQN == null) {
      typeClassSuperClassesFQN = typeClass.superClasses().stream().map(Symbol::fullyQualifiedName).collect(Collectors.toSet());
    }
    return typeClassSuperClassesFQN;
  }

  private Set<String> typeClassMembersFQN() {
    if (typeClassMembersFQN == null) {
      typeClassMembersFQN = typeClass.declaredMembers().stream().map(Symbol::fullyQualifiedName).collect(Collectors.toSet());
    }
    return typeClassMembersFQN;
  }

  boolean hasUnresolvedHierarchy() {
    return ((ClassSymbolImpl) typeClass).hasUnresolvedTypeHierarchy(false);
  }

  @Override
  public int hashCode() {
    return Objects.hash(typeClass.name(), typeClass.fullyQualifiedName(), typeClassSuperClassesFQN(), typeClassMembersFQN());
  }

  @Override
  public String toString() {
    return "RuntimeType(" + typeClass.fullyQualifiedName() + ')';
  }

  public ClassSymbol getTypeClass() {
    return typeClass;
  }
}
