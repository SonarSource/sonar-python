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
package org.sonar.python.types.v3;

import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.UnknownType;
import org.sonar.python.types.v3.TypeCheckerPoc.ClassTypeBuilder;
import org.sonar.python.types.v3.TypeCheckerPoc.InnerPredicate;
import org.sonar.python.types.v3.TypeCheckerPoc.InnerPredicateBuilder;
import org.sonar.python.types.v3.TypeCheckerPoc.ObjectTypeBuilder;
import org.sonar.python.types.v3.TypeCheckerPoc.UnspecializedTypeCheckerBuilder;

public class TypeCheckerPocPredicates {
  // METHODS
  static InnerPredicateBuilder<UnspecializedTypeCheckerBuilder, ClassTypeBuilder> isClass() {
    return (builder, ctx) -> {
      builder.addPredicate(new IsClassTypeInnerPredicate());
      return new ClassTypeBuilder(builder);
    };
  }

  private static class IsClassTypeInnerPredicate implements InnerPredicate {
    @Override
    public TriBool apply(PythonType type) {
      return type instanceof ClassType ? TriBool.TRUE : TriBool.FALSE;
    }
  }

  static InnerPredicateBuilder<UnspecializedTypeCheckerBuilder, ObjectTypeBuilder> isObject(String fqn) {
    return (builder, ctx) -> {
      PythonType resolvedType = ctx.getProjectLevelTypeTable().getType(fqn);
      if (resolvedType instanceof UnknownType.UnknownTypeImpl) {
        throw new IllegalStateException("Tried to match UnknownType");
      }
      builder.addPredicate(new ObjectIsInnerPredicate(resolvedType));
      return new ObjectTypeBuilder(builder);
    };
  }

  private static class ObjectIsInnerPredicate implements InnerPredicate {

    private final PythonType resolvedType;

    private ObjectIsInnerPredicate(PythonType resolvedType) {
      this.resolvedType = resolvedType;
    }

    @Override
    public TriBool apply(PythonType type) {
      if (!(type instanceof ObjectType)) {
        return TriBool.FALSE;
      }
      type = type.unwrappedType();
      if (type instanceof UnknownType.UnresolvedImportType unresolvedPythonType && resolvedType instanceof UnknownType.UnresolvedImportType unresolvedExpectedType) {
        return unresolvedPythonType.importPath().equals(unresolvedExpectedType.importPath()) ? TriBool.TRUE : TriBool.UNKNOWN;
      }
      if (type instanceof UnknownType || resolvedType instanceof UnknownType) {
        return TriBool.UNKNOWN;
      }
      return type.equals(resolvedType) ? TriBool.TRUE : TriBool.FALSE;
    }
  }

}
