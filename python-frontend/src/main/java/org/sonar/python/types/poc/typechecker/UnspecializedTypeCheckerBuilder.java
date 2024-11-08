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
package org.sonar.python.types.poc.typechecker;

import org.sonar.python.types.poc.AbstractTypeCheckerBuilder;
import org.sonar.python.types.poc.InnerPredicate;
import org.sonar.python.types.poc.TypeCheckerBuilderContext;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.UnknownType;

public class UnspecializedTypeCheckerBuilder extends AbstractTypeCheckerBuilder<UnspecializedTypeCheckerBuilder> {
  public UnspecializedTypeCheckerBuilder(TypeCheckerBuilderContext projectLevelTypeTable) {
    super(projectLevelTypeTable);
  }

  @Override
  public UnspecializedTypeCheckerBuilder rebind(TypeCheckerBuilderContext context) {
    return new UnspecializedTypeCheckerBuilder(context);
  }

  public ClassTypeBuilder isClass() {
    getContext().addPredicate(new IsClassPredicate());
    return new ClassTypeBuilder(getContext());
  }

  private static class IsClassPredicate implements InnerPredicate {
    @Override
    public TriBool applyOn(PythonType type) {
      return type instanceof ClassType ? TriBool.TRUE : TriBool.FALSE;
    }
  }

  public ObjectTypeBuilder isObject(String fqn) {
    PythonType resolvedType = getContext().getProjectLevelTypeTable().getType(fqn);
    if (resolvedType instanceof UnknownType.UnknownTypeImpl) {
      throw new IllegalStateException("Tried to match UnknownType");
    }
    getContext().addPredicate(new ObjectIsInnerPredicate(resolvedType));
    return new ObjectTypeBuilder(getContext());
  }

  private record ObjectIsInnerPredicate(PythonType resolvedType) implements InnerPredicate {

    @Override
    public TriBool applyOn(PythonType type) {
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
