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
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;

public class TypeCheckBuilder {

  ProjectLevelTypeTable projectLevelTypeTable;
  TypePredicate predicate = new AndTypePredicate(new ArrayList<>());

  public TypeCheckBuilder(ProjectLevelTypeTable projectLevelTypeTable) {
    this.projectLevelTypeTable = projectLevelTypeTable;
  }

  public TypeCheckBuilder hasMember(String memberName) {
    this.predicate = predicate.and(new HasMemberTypePredicate(memberName));
    return this;
  }

  public TypeCheckBuilder instancesHaveMember(String memberName) {
    this.predicate = predicate.and(new InstancesHaveMemberTypePredicate(memberName));
    return this;
  }

  public TypeCheckBuilder isTypeHintTypeSource() {
    this.predicate = predicate.and(new TypeSourceMatcherTypePredicate(TypeSource.TYPE_HINT));
    return this;
  }

  public TypeCheckBuilder isBuiltinWithName(String name) {
    PythonType builtinType = projectLevelTypeTable.getModule().resolveMember(name).orElse(PythonType.UNKNOWN);
    this.predicate = predicate.and(new IsSameAsTypePredicate(builtinType));
    return this;
  }

  /**
   * By default, chaining predicates in the TypeCheckBuilder will AND the consecutive predicates
   * This method allows to OR them instead
   */
  public TypeCheckBuilder or(TypeCheckBuilder typeCheckBuilder) {
    this.predicate = predicate.or(typeCheckBuilder.predicate);
    return this;
  }

  public TriBool check(PythonType pythonType) {
    return predicate.test(pythonType);
  }

  interface TypePredicate {
    TriBool test(PythonType pythonType);

    default TypePredicate and(TypePredicate typePredicate) {
      if (this instanceof AndTypePredicate andTypePredicate) {
        andTypePredicate.and(typePredicate);
        return this;
      }
      List<TypePredicate> andedPredicates = new ArrayList<>();
      andedPredicates.add(this);
      andedPredicates.add(typePredicate);
      return new AndTypePredicate(andedPredicates);
    }

    default TypePredicate or(TypePredicate typePredicate) {
      if (this instanceof OrTypePredicate orTypePredicate) {
        orTypePredicate.and(typePredicate);
        return this;
      }
      List<TypePredicate> oredPredicates = new ArrayList<>();
      oredPredicates.add(this);
      oredPredicates.add(typePredicate);
      return new OrTypePredicate(oredPredicates);
    }
  }

  static class AndTypePredicate implements TypePredicate {

    List<TypePredicate> andedPredicates;

    public AndTypePredicate(List<TypePredicate> andedPredicates) {
      this.andedPredicates = andedPredicates;
    }

    @Override
    public TypePredicate and(TypePredicate typePredicate) {
      andedPredicates.add(typePredicate);
      return this;
    }

    public TriBool test(PythonType pythonType) {
      TriBool result = TriBool.TRUE;
      for (TypePredicate predicate : andedPredicates) {
        TriBool partialResult = predicate.test(pythonType);
        result = result.and(partialResult);
        if (result == TriBool.UNKNOWN) {
          return TriBool.UNKNOWN;
        }
      }
      return result;
    }
  }

  static class OrTypePredicate implements TypePredicate {

    List<TypePredicate> oredPredicates;

    public OrTypePredicate(List<TypePredicate> andedPredicates) {
      this.oredPredicates = andedPredicates;
    }

    @Override
    public TypePredicate or(TypePredicate typePredicate) {
      oredPredicates.add(typePredicate);
      return this;
    }

    public TriBool test(PythonType pythonType) {
      TriBool result = TriBool.FALSE;
      for (TypePredicate predicate : oredPredicates) {
        TriBool partialResult = predicate.test(pythonType);
        result = result.or(partialResult);
        if (result == TriBool.TRUE) {
          return TriBool.TRUE;
        }
      }
      return result;
    }
  }

  static class HasMemberTypePredicate implements TypePredicate {
    String memberName;

    public HasMemberTypePredicate(String memberName) {
      this.memberName = memberName;
    }

    @Override
    public TriBool test(PythonType pythonType) {
      return pythonType.hasMember(memberName);
    }
  }

  static class InstancesHaveMemberTypePredicate implements TypePredicate {
    String memberName;

    public InstancesHaveMemberTypePredicate(String memberName) {
      this.memberName = memberName;
    }

    @Override
    public TriBool test(PythonType pythonType) {
      if (pythonType instanceof ClassType classType) {
        return classType.instancesHaveMember(memberName);
      }
      return TriBool.FALSE;
    }
  }

  record TypeSourceMatcherTypePredicate(TypeSource typeSource) implements TypePredicate {

    @Override
    public TriBool test(PythonType pythonType) {
      return pythonType.typeSource() == typeSource ? TriBool.TRUE : TriBool.FALSE;
    }
  }

  static class IsSameAsTypePredicate implements TypePredicate {

    PythonType expectedType;

    public IsSameAsTypePredicate(PythonType expectedType) {
      this.expectedType = expectedType;
    }

    @Override
    public TriBool test(PythonType pythonType) {
      if (pythonType instanceof ObjectType objectType) {
        pythonType = objectType.unwrappedType();
      }
      if (pythonType == PythonType.UNKNOWN || expectedType == PythonType.UNKNOWN) {
        return TriBool.UNKNOWN;
      }
      return pythonType.equals(expectedType) ? TriBool.TRUE : TriBool.FALSE;
    }
  }
}
