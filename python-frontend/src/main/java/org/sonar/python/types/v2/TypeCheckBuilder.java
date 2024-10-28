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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.sonar.python.semantic.v2.TypeTable;

public class TypeCheckBuilder {

  TypeTable projectLevelTypeTable;
  List<TypePredicate> predicates = new ArrayList<>();

  public TypeCheckBuilder(TypeTable projectLevelTypeTable) {
    this.projectLevelTypeTable = projectLevelTypeTable;
  }

  public TypeCheckBuilder hasMember(String memberName) {
    predicates.add(new HasMemberTypePredicate(memberName));
    return this;
  }

  public TypeCheckBuilder instancesHaveMember(String memberName) {
    predicates.add(new InstancesHaveMemberTypePredicate(memberName));
    return this;
  }

  public TypeCheckBuilder isTypeHintTypeSource() {
    predicates.add(new TypeSourceMatcherTypePredicate(TypeSource.TYPE_HINT));
    return this;
  }

  public TypeCheckBuilder isExactTypeSource() {
    predicates.add(new TypeSourceMatcherTypePredicate(TypeSource.EXACT));
    return this;
  }

  public TypeCheckBuilder isBuiltinWithName(String name) {
    PythonType builtinType = projectLevelTypeTable.getBuiltinsModule().resolveMember(name).orElse(PythonType.UNKNOWN);
    predicates.add(new IsSameAsTypePredicate(builtinType));
    return this;
  }

  public TriBool check(PythonType pythonType) {
    TriBool result = TriBool.TRUE;
    for (TypePredicate predicate : predicates) {
      TriBool partialResult = predicate.test(pythonType);
      result = result.and(partialResult);
      if (result == TriBool.UNKNOWN) {
        return TriBool.UNKNOWN;
      }
    }
    return result;
  }

  public TypeCheckBuilder isInstanceOf(String fqn) {
    var expected = projectLevelTypeTable.getType(fqn);
    predicates.add(new IsInstanceOfPredicate(expected));
    return this;
  }

  public TypeCheckBuilder isTypeOrInstanceWithName(String expectedName) {
    var expected = projectLevelTypeTable.getType(expectedName);
    predicates.add(new IsSameAsTypePredicate(expected, false));
    return this;
  }

  public TypeCheckBuilder isTypeWithName(String expectedName) {
    var expected = projectLevelTypeTable.getType(expectedName);
    predicates.add(new IsSameAsTypePredicate(expected, true));
    return this;
  }

  interface TypePredicate {
    TriBool test(PythonType pythonType);
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
    boolean isStrictCheck;

    public IsSameAsTypePredicate(PythonType expectedType) {
      this(expectedType, false);
    }

    public IsSameAsTypePredicate(PythonType expectedType, boolean isStrictCheck) {
      this.expectedType = expectedType;
      this.isStrictCheck = isStrictCheck;
    }

    @Override
    public TriBool test(PythonType pythonType) {
      if ((pythonType instanceof ObjectType objectType) && !isStrictCheck) {
        pythonType = objectType.unwrappedType();
      }
      if (pythonType instanceof UnknownType.UnresolvedImportType unresolvedPythonType && expectedType instanceof UnknownType.UnresolvedImportType unresolvedExpectedType) {
        return unresolvedPythonType.importPath().equals(unresolvedExpectedType.importPath()) ? TriBool.TRUE : TriBool.UNKNOWN;
      }
      if (pythonType instanceof UnknownType || expectedType instanceof UnknownType) {
        return TriBool.UNKNOWN;
      }
      return pythonType.equals(expectedType) ? TriBool.TRUE : TriBool.FALSE;
    }
  }

  record IsInstanceOfPredicate(PythonType expectedType)  implements TypePredicate {

    @Override
    public TriBool test(PythonType pythonType) {
      if (expectedType instanceof ClassType expectedClassType) {
        if (pythonType instanceof ObjectType objectType) {
          if (objectType.type() instanceof ClassType classType) {
            // when the checking type is an ObjectType of a ClassType
            return isClassInheritedFrom(classType, expectedClassType);
          } else if (objectType.type() instanceof UnionType unionType) {
            // when the checking type is an ObjectType of a UnionType
            return isObjectOfUnionTypeInstanceOf(expectedClassType, unionType);
          }
        } else if (pythonType instanceof UnionType unionType) {
          // when the checking type is a UnionType
          return isUnionTypeInstanceOf(unionType);
        }
      }
      return TriBool.UNKNOWN;
    }

    private static TriBool isObjectOfUnionTypeInstanceOf(ClassType expectedClassType, UnionType unionType) {
      var results = unionType.candidates()
        .stream()
        .map(classType -> isClassInheritedFrom(classType, expectedClassType))
        .distinct()
        .toList();

      if (results.size() > 1) {
        return TriBool.UNKNOWN;
      } else {
        return results.get(0);
      }
    }

    private TriBool isUnionTypeInstanceOf(UnionType unionType) {
      var candidatesResults = unionType.candidates()
        .stream()
        .map(this::test)
        .distinct()
        .toList();

      if (candidatesResults.size() != 1) {
        return TriBool.UNKNOWN;
      } else {
        return candidatesResults.get(0);
      }
    }

    private static TriBool isClassInheritedFrom(PythonType classType, ClassType expectedClassType) {
      if (classType == expectedClassType) {
        return TriBool.TRUE;
      }

      var types = collectTypes(classType);

      if (types.contains(expectedClassType)) {
        return TriBool.TRUE;
      } else if (containsUnknown(types)) {
        return TriBool.UNKNOWN;
      } else {
        return TriBool.FALSE;
      }
    }

    private static Set<PythonType> collectTypes(PythonType type) {
      var result = new HashSet<PythonType>();
      var queue = new ArrayDeque<PythonType>();
      queue.add(type);
      while (!queue.isEmpty()) {
        var currentType = queue.pop();
        if (result.contains(currentType)) {
          continue;
        }
        result.add(currentType);
        if (currentType instanceof UnionType) {
          result.clear();
          result.add(PythonType.UNKNOWN);
          queue.clear();
        } else if (currentType instanceof ClassType classType) {
          queue.addAll(classType.superClasses().stream().map(TypeWrapper::type).toList());
        }
      }
      return result;
    }
  }

  private static boolean containsUnknown(Set<PythonType> types) {
    return types.stream().anyMatch(UnknownType.class::isInstance);
  }
}
