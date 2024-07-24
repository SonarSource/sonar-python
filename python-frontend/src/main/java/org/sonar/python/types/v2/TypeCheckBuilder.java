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

  TypeCheckBuilder(ProjectLevelTypeTable projectLevelTypeTable, TypePredicate predicate) {
    this.projectLevelTypeTable = projectLevelTypeTable;
    this.predicate = predicate;
  }

  public TypeCheckBuilder hasMember(String memberName) {
    return this.and(new TypeCheckBuilder(projectLevelTypeTable, new HasMemberTypePredicate(memberName)));
  }

  public TypeCheckBuilder instancesHaveMember(String memberName) {
    return this.and(new TypeCheckBuilder(projectLevelTypeTable, new InstancesHaveMemberTypePredicate(memberName)));
  }

  public TypeCheckBuilder isTypeHintTypeSource() {
    return this.and(new TypeCheckBuilder(projectLevelTypeTable, new TypeSourceMatcherTypePredicate(TypeSource.TYPE_HINT)));
  }

  public TypeCheckBuilder isBuiltinWithName(String name) {
    PythonType builtinType = projectLevelTypeTable.getModule().resolveMember(name).orElse(PythonType.UNKNOWN);
    return this.and(new TypeCheckBuilder(projectLevelTypeTable, new IsSameAsTypePredicate(builtinType)));
  }

  /**
   * By default, chaining predicates in the TypeCheckBuilder will AND the consecutive predicates
   * This method allows to OR them instead
   */
  TypeCheckBuilder or(TypeCheckBuilder typeCheckBuilder) {
    return new TypeCheckBuilder(projectLevelTypeTable, predicate.or(typeCheckBuilder.predicate));
  }

  TypeCheckBuilder and(TypeCheckBuilder typeCheckBuilder) {
    return new TypeCheckBuilder(projectLevelTypeTable, predicate.and(typeCheckBuilder.predicate));
  }

  public TriBool check(PythonType pythonType) {
    return predicate.test(pythonType);
  }

  interface TypePredicate {
    TriBool test(PythonType pythonType);

    default TypePredicate and(TypePredicate typePredicate) {
      return new AndTypePredicate(this, typePredicate);
    }

    default TypePredicate or(TypePredicate typePredicate) {
      return new OrTypePredicate(this, typePredicate);
    }
  }

  static class AndTypePredicate implements TypePredicate {

    List<TypePredicate> andedPredicates;

    public AndTypePredicate(List<TypePredicate> andedPredicates) {
      this.andedPredicates = andedPredicates;
    }

    public AndTypePredicate(TypePredicate firstPredicate, TypePredicate secondPredicate) {
      List<TypePredicate> unwrapped = new ArrayList<>();
      if (firstPredicate instanceof  AndTypePredicate andTypePredicate) {
        unwrapped.addAll(andTypePredicate.andedPredicates);
      } else {
        unwrapped.add(firstPredicate);
      }
      if (secondPredicate instanceof  AndTypePredicate andTypePredicate) {
        unwrapped.addAll(andTypePredicate.andedPredicates);
      } else {
        unwrapped.add(secondPredicate);
      }
      this.andedPredicates = unwrapped;
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

    public OrTypePredicate(TypePredicate firstPredicate, TypePredicate secondPredicate) {
      List<TypePredicate> unwrapped = new ArrayList<>();
      if (firstPredicate instanceof  OrTypePredicate orTypePredicate) {
        unwrapped.addAll(orTypePredicate.oredPredicates);
      } else {
        unwrapped.add(firstPredicate);
      }
      if (secondPredicate instanceof  OrTypePredicate orTypePredicate) {
        unwrapped.addAll(orTypePredicate.oredPredicates);
      } else {
        unwrapped.add(secondPredicate);
      }
      this.oredPredicates = unwrapped;
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
