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

import java.util.LinkedHashSet;
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
    for (ClassSymbol classSymbol : classesToExplore()) {
      for (Symbol member : classSymbol.declaredMembers()) {
        if (member.name().equals(memberName)) {
          return true;
        }
      }
    }
    return false;
  }

  @Override
  public Optional<Symbol> resolveMember(String memberName) {
    LinkedHashSet<ClassSymbol> classSymbols = classesToExplore();
    for (ClassSymbol classSymbol : classSymbols) {
      for (Symbol member : classSymbol.declaredMembers()) {
        if (member.name().equals(memberName)) {
          return Optional.of(member);
        }
      }
    }
    return Optional.empty();
  }

  @Override
  public boolean canOnlyBe(String typeName) {
    return typeName.equals(typeClass.fullyQualifiedName());
  }

  @Override
  public boolean canBeOrExtend(String typeName) {
    return classesToExplore().stream().anyMatch(
      classSymbol -> typeName.equals(classSymbol.fullyQualifiedName()) || classSymbol.hasUnresolvedTypeHierarchy());
  }

  private LinkedHashSet<ClassSymbol> classesToExplore() {
    LinkedHashSet<ClassSymbol> set = new LinkedHashSet<>();
    addClassesToExplore(typeClass, set);
    return set;
  }

  private static void addClassesToExplore(ClassSymbol typeClass, LinkedHashSet<ClassSymbol> set) {
    if (set.add(typeClass)) {
      for (Symbol superClass : typeClass.superClasses()) {
        if (superClass instanceof ClassSymbol) {
          addClassesToExplore((ClassSymbol) superClass, set);
        }
      }
    }
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
