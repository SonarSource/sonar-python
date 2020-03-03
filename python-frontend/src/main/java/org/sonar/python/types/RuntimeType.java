/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.types.InferredType;

class RuntimeType implements InferredType {

  private final ClassSymbol typeClass;

  RuntimeType(ClassSymbol typeClass) {
    this.typeClass = typeClass;
  }

  @Override
  public boolean isIdentityComparableWith(InferredType other) {
    if (other == AnyType.ANY) {
      return true;
    }
    if (other instanceof UnionType) {
      return other.isIdentityComparableWith(this);
    }
    return this.equals(other);
  }

  @Override
  public boolean canHaveMember(String memberName) {
    if (typeClass.hasUnresolvedTypeHierarchy()) {
      return true;
    }
    return typeClass.resolveMember(memberName).isPresent();
  }

  @Override
  public Optional<Symbol> resolveMember(String memberName) {
    return typeClass.resolveMember(memberName);
  }

  @Override
  public boolean canOnlyBe(String typeName) {
    return typeName.equals(typeClass.fullyQualifiedName());
  }

  @Override
  public boolean canBeOrExtend(String typeName) {
    return typeClass.isOrExtends(typeName) || typeClass.hasUnresolvedTypeHierarchy();
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
    return Objects.equals(typeClass.fullyQualifiedName(), that.typeClass.fullyQualifiedName());
  }

  @Override
  public int hashCode() {
    return Objects.hash(typeClass.fullyQualifiedName());
  }

  @Override
  public String toString() {
    return "RuntimeType(" + typeClass.fullyQualifiedName() + ')';
  }
}
