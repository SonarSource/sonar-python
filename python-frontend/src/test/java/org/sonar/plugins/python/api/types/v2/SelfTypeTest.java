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
package org.sonar.plugins.python.api.types.v2;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.TypesTestUtils.BOOL_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.STR_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;

class SelfTypeTest {

  // Core functionality (creation, display names, toString, delegation)
  @Test
  void basicSelfType() {
    ClassType classType = (ClassType) INT_TYPE;
    
    PythonType selfType = SelfType.of(classType);
    
    assertThat(selfType).isInstanceOf(SelfType.class);
    SelfType actualSelfType = (SelfType) selfType;
    assertThat(actualSelfType.innerType()).isEqualTo(classType);
    assertThat(actualSelfType.unwrappedType()).isEqualTo(classType.unwrappedType());
  }

  @Test
  void selfTypeDisplayName() {
    ClassType classType = (ClassType) INT_TYPE;
    
    SelfType selfType = (SelfType) SelfType.of(classType);
    
    assertThat(selfType.displayName()).isPresent();
    assertThat(selfType.instanceDisplayName()).isPresent();
    assertThat(selfType.name()).contains("Self[");
    assertThat(selfType.key()).contains("Self[");
  }

  @Test
  void selfTypeCompatibility() {
    ClassType classTypeA = (ClassType) INT_TYPE;
    ClassType classTypeB = (ClassType) STR_TYPE;
    
    SelfType selfTypeA = (SelfType) SelfType.of(classTypeA);
    SelfType selfTypeB = (SelfType) SelfType.of(classTypeB);
    
    assertThat(selfTypeA.isCompatibleWith(selfTypeA)).isTrue();
    assertThat(selfTypeA.isCompatibleWith(classTypeA)).isTrue();
    assertThat(selfTypeA.isCompatibleWith(selfTypeB)).isFalse();
    assertThat(selfTypeA.isCompatibleWith(classTypeB)).isFalse();
  }

  @Test
  void selfTypeMemberResolution() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        def foo(self): ...
        def bar(self): ...
      """);
    ClassType classType = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    
    SelfType selfType = (SelfType) SelfType.of(classType);
    
    assertThat(selfType.hasMember("foo")).isEqualTo(TriBool.TRUE);
    assertThat(selfType.hasMember("bar")).isEqualTo(TriBool.TRUE);
    assertThat(selfType.hasMember("baz")).isIn(TriBool.FALSE, TriBool.UNKNOWN);
    
    assertThat(selfType.resolveMember("foo")).isPresent();
    assertThat(selfType.resolveMember("bar")).isPresent();
    assertThat(selfType.resolveMember("baz")).isEmpty();
  }

  @Test
  void selfTypeDefinitionLocation() {
    FileInput fileInput = parseAndInferTypes("""
      class A: ...
      """);
    ClassType classType = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    
    SelfType selfType = (SelfType) SelfType.of(classType);
    
    assertThat(selfType.definitionLocation()).isPresent();
    assertThat(selfType.definitionLocation()).isEqualTo(classType.definitionLocation());
  }

  @Test
  void selfTypeEquality() {
    ClassType classType = (ClassType) INT_TYPE;
    
    SelfType selfType1 = (SelfType) SelfType.of(classType);
    SelfType selfType2 = (SelfType) SelfType.of(classType);
    
    assertThat(selfType1)
      .isEqualTo(selfType1)
      .isEqualTo(selfType2)
      .hasSameHashCodeAs(selfType2);
    
    assertThat(selfType1.equals(null)).isFalse();

    assertThat(selfType1.equals(classType)).isFalse();
  }

  @Test
  void selfTypeTypeSource() {
    ClassType classType = (ClassType) INT_TYPE;
    
    SelfType selfType = (SelfType) SelfType.of(classType);
    
    assertThat(selfType.typeSource()).isEqualTo(classType.typeSource());
  }

  @Test
  void selfTypeToString() {
    ClassType classType = (ClassType) INT_TYPE;
    SelfType selfType = (SelfType) SelfType.of(classType);
    
    assertThat(selfType.toString()).contains("SelfType[").contains(classType.toString());
  }

  // Edge cases (null, unknown)
  @Test
  void selfTypeOfNull() {
    PythonType result = SelfType.of(null);
    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void selfTypeOfUnknown() {
    PythonType result = SelfType.of(PythonType.UNKNOWN);
    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  // Critical invariants (no double-wrapping, proper ObjectType/UnionType handling)
  @Test
  void selfTypeOfSelfType() {
    ClassType classType = (ClassType) INT_TYPE;
    
    SelfType selfType = (SelfType) SelfType.of(classType);
    PythonType selfTypeOfSelfType = SelfType.of(selfType);
    
    assertThat(selfTypeOfSelfType).isSameAs(selfType);
  }

  @Test
  void selfTypeOfObjectTypeWithAttributes() {
    ClassType classType = (ClassType) BOOL_TYPE;
    List<PythonType> attributes = List.of(INT_TYPE, STR_TYPE);
    List<Member> members = List.of(
      new Member("foo", new FunctionTypeBuilder("foo").build())
    );
    ObjectType objectType = ObjectType.Builder.fromType(classType)
      .withAttributes(attributes)
      .withMembers(members)
      .build();
    
    PythonType result = SelfType.of(objectType);
    
    // Should return ObjectType[SelfType[ClassType]], not SelfType[ObjectType[ClassType]]
    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType resultObjectType = (ObjectType) result;
    
    // Should preserve attributes and members
    assertThat(resultObjectType.attributes()).isEqualTo(attributes);
    assertThat(resultObjectType.members()).isEqualTo(members);
    
    assertThat(resultObjectType.unwrappedType()).isInstanceOf(SelfType.class);
    SelfType innerSelfType = (SelfType) resultObjectType.unwrappedType();
    assertThat(innerSelfType.innerType()).isEqualTo(classType);
  }

  @Test
  void selfTypeOfUnionType() {
    ClassType classTypeA = (ClassType) INT_TYPE;
    ClassType classTypeB = (ClassType) STR_TYPE;
    
    UnionType unionType = (UnionType) UnionType.or(classTypeA, classTypeB);
    
    PythonType result = SelfType.of(unionType);
    
    // Should return Union[Self[A], Self[B]], not Self[Union[A, B]]
    assertThat(result).isInstanceOf(UnionType.class);
    UnionType resultUnionType = (UnionType) result;
    
    assertThat(resultUnionType.candidates()).hasSize(2);
    assertThat(resultUnionType.candidates()).allMatch(SelfType.class::isInstance);
    
    Set<PythonType> innerTypes = resultUnionType.candidates().stream()
      .map(SelfType.class::cast)
      .map(SelfType::innerType)
      .collect(java.util.stream.Collectors.toSet());
    
    assertThat(innerTypes).containsExactlyInAnyOrder(classTypeA, classTypeB);
  }

  @Test
  void selfTypeOfUnsupportedTypeReturnsUnknown() {
    FunctionType functionType = new FunctionTypeBuilder("myFunction").build();

    PythonType result = SelfType.of(functionType);

    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void selfTypeOfUnionWithNonClassTypeReturnsUnknown() {
    ClassType classType = (ClassType) INT_TYPE;
    FunctionType functionType = new FunctionTypeBuilder("myFunction").build();

    UnionType unionType = (UnionType) UnionType.or(classType, functionType);

    PythonType result = SelfType.of(unionType);

    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void selfTypeOfObjectTypeWrappingUnionOfClassTypes() {
    ClassType classTypeA = (ClassType) INT_TYPE;
    ClassType classTypeB = (ClassType) STR_TYPE;
    UnionType unionType = (UnionType) UnionType.or(classTypeA, classTypeB);

    ObjectType objectType = ObjectType.Builder.fromType(classTypeA)
      .withType(unionType)
      .build();

    PythonType result = SelfType.of(objectType);

    // Should return ObjectType[Union[Self[A], Self[B]]]
    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType resultObjectType = (ObjectType) result;

    assertThat(resultObjectType.unwrappedType()).isInstanceOf(UnionType.class);
    UnionType resultUnionType = (UnionType) resultObjectType.unwrappedType();

    assertThat(resultUnionType.candidates()).hasSize(2);
    assertThat(resultUnionType.candidates()).allMatch(SelfType.class::isInstance);

    Set<PythonType> innerTypes = resultUnionType.candidates().stream()
      .map(SelfType.class::cast)
      .map(SelfType::innerType)
      .collect(java.util.stream.Collectors.toSet());

    assertThat(innerTypes).containsExactlyInAnyOrder(classTypeA, classTypeB);
  }

  @Test
  void selfTypeOfObjectTypeWrappingUnionWithNonClassTypeReturnsUnknown() {
    ClassType classType = (ClassType) INT_TYPE;
    FunctionType functionType = new FunctionTypeBuilder("myFunction").build();
    UnionType unionType = (UnionType) UnionType.or(classType, functionType);

    ObjectType objectType = ObjectType.Builder.fromType(classType)
      .withType(unionType)
      .build();

    PythonType result = SelfType.of(objectType);

    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

}
