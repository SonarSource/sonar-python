/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.types.v2;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.semantic.v2.typetable.TypeTable;

import static org.sonar.python.types.v2.TypeUtils.collectTypes;

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

  public TypeCheckBuilder isSubtypeOf(String fqn) {
    PythonType type = projectLevelTypeTable.getType(fqn);
    predicates.add(new IsSubtypeOfPredicate(type, fqn));
    return this;
  }

  public TypeCheckBuilder isInstance() {
    predicates.add(new IsInstancePredicate());
    return this;
  }

  public TypeCheckBuilder isGeneric() {
    predicates.add(new IsGenericPredicate());
    return this;
  }

  public TriBool check(PythonType pythonType) {
    TriBool result = TriBool.TRUE;
    for (TypePredicate predicate : predicates) {
      TriBool partialResult = predicate.test(pythonType);
      result = result.conservativeAnd(partialResult);
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

  public TypeCheckBuilder isTypeWithFqn(String expectedFqn) {
    predicates.add(new IsTypeWithFullyQualifiedNamePredicate(expectedFqn));
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

  static class IsTypeWithFullyQualifiedNamePredicate implements TypePredicate {
    String expectedFullyQualifiedName;

    public IsTypeWithFullyQualifiedNamePredicate(String expectedFullyQualifiedName) {
      this.expectedFullyQualifiedName = expectedFullyQualifiedName;
    }

    @Override
    public TriBool test(PythonType pythonType) {
      return Optional.of(pythonType)
        .map(IsTypeWithFullyQualifiedNamePredicate::getFullyQualifiedName)
        .map(typeFqn -> Objects.equals(expectedFullyQualifiedName, typeFqn) ? TriBool.TRUE : TriBool.FALSE)
        .orElse(TriBool.UNKNOWN);
    }

    @CheckForNull
    private static String getFullyQualifiedName(PythonType type) {
      if (type instanceof FunctionType functionType) {
        return functionType.fullyQualifiedName();
      } else if (type instanceof ClassType classType) {
        return classType.fullyQualifiedName();
      } else if (type instanceof UnknownType.UnresolvedImportType unresolvedImportType) {
        return unresolvedImportType.importPath();
      }
      return null;
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
      if (pythonType instanceof SelfType || expectedType instanceof SelfType) {
        return TriBool.UNKNOWN;
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

  record IsSubtypeOfPredicate(PythonType expectedType, String expectedFqnName) implements TypePredicate {

    @Override
    public TriBool test(PythonType pythonType) {
      if (pythonType instanceof ClassType testedClassType) {
        // SONARPY-2593: We shouldn't have to rely on fully qualified names here
        var types = collectTypes(testedClassType);
        if (types.stream().anyMatch(t -> (t instanceof ClassType ct) && ct.fullyQualifiedName().equals(expectedFqnName))) {
          return TriBool.TRUE;
        }
        if (expectedType instanceof UnknownType && types.stream().anyMatch(t -> new IsTypeWithFullyQualifiedNamePredicate(expectedFqnName).test(t).isTrue())) {
          return TriBool.TRUE;
        }
      }
      if (expectedType instanceof ClassType expectedClassType) {
        return isClassInheritedFrom(pythonType, expectedClassType);
      }
      return TriBool.UNKNOWN;
    }
  }

  record IsInstanceOfPredicate(PythonType expectedType) implements TypePredicate {

    @Override
    public TriBool test(PythonType pythonType) {
      if (pythonType instanceof SelfType || expectedType instanceof SelfType) {
        return TriBool.UNKNOWN;
      }
      if (expectedType instanceof ClassType expectedClassType) {
        if (pythonType instanceof ObjectType objectType) {
          PythonType unwrappedType = objectType.type();
          if (unwrappedType instanceof ClassType classType) {
            // when the checking type is an ObjectType of a ClassType
            return isClassInheritedFrom(classType, expectedClassType);
          } else if (unwrappedType instanceof UnionType unionType) {
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
  }

  record IsInstancePredicate() implements TypePredicate {

    @Override
    public TriBool test(PythonType pythonType) {
      Set<PythonType> candidates = Set.of(pythonType);
      if (pythonType instanceof UnionType unionType) {
        candidates = unionType.candidates();
      }
      if (candidates.stream().allMatch(ObjectType.class::isInstance)) {
        return TriBool.TRUE;
      }
      return TriBool.FALSE;
    }
  }

  record IsGenericPredicate() implements TypePredicate {

    @Override
    public TriBool test(PythonType pythonType) {
      return isGeneric(pythonType);
    }

    private static TriBool isGeneric(PythonType pythonType) {
      if (pythonType instanceof ClassType classType && classType.isGeneric()) {
        return TriBool.TRUE;
      }
      if (pythonType instanceof SpecialFormType specialFormType) {
        return "typing.Generic".equals(specialFormType.fullyQualifiedName()) ? TriBool.TRUE : TriBool.UNKNOWN;
      }
      if (pythonType instanceof UnknownType.UnresolvedImportType unresolvedImportType && "typing.Generic".equals(unresolvedImportType.importPath())) {
        return TriBool.TRUE;
      }
      return TriBool.UNKNOWN;
    }
  }

  private static TriBool isClassInheritedFrom(PythonType classType, ClassType expectedParentClassType) {
    if (classType == expectedParentClassType) {
      return TriBool.TRUE;
    }

    var types = collectTypes(classType);

    if (types.contains(expectedParentClassType)) {
      return TriBool.TRUE;
    } else if (containsUnknown(types)) {
      return TriBool.UNKNOWN;
    } else {
      return TriBool.FALSE;
    }
  }



  private static boolean containsUnknown(Set<PythonType> types) {
    return types.stream().anyMatch(UnknownType.class::isInstance);
  }
}
