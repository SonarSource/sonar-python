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

import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.stream.Stream;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.ObjectTypeBuilder;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.types.v2.UnknownType.UnresolvedImportType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

class TypeCheckerBuilderTest {

  @Test
  void typeSourceTest() {
    var builder = new TypeCheckBuilder(null).isTypeHintTypeSource();
    assertThat(builder.check(new ObjectType(PythonType.UNKNOWN, TypeSource.TYPE_HINT)))
      .isEqualTo(TriBool.TRUE);
    assertThat(builder.check(new ObjectType(PythonType.UNKNOWN, TypeSource.EXACT)))
      .isEqualTo(TriBool.FALSE);
  }

  static Stream<Arguments> isInstanceOfTestProvider() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    var isIntBuilder = new TypeCheckBuilder(table).isInstanceOf("int");

    var intClassType = table.getType("int");
    var strClassType = table.getType("str");

    var aClassType = new ClassTypeBuilder()
      .withName("A")
      .withSuperClasses(intClassType)
      .build();

    var bClassType = new ClassTypeBuilder()
      .withName("B")
      .withSuperClasses(strClassType)
      .build();

    var cClassType = new ClassTypeBuilder()
      .withName("C")
      .withSuperClasses(PythonType.UNKNOWN, intClassType)
      .build();

    var dClassType = new ClassTypeBuilder()
      .withName("D")
      .withSuperClasses(PythonType.UNKNOWN, strClassType)
      .build();

    var iClassType = new ClassTypeBuilder()
      .withName("I")
      .withSuperClasses(aClassType, intClassType)
      .build();

    var jClassType = new ClassTypeBuilder()
      .withName("J")
      .withSuperClasses(UnionType.or(bClassType, aClassType))
      .build();

    var unresolvedType = new UnresolvedImportType("unknown");

    var intObjectType = new ObjectType(intClassType);
    var strObjectType = new ObjectType(strClassType);
    var aObject = new ObjectType(aClassType);
    var bObject = new ObjectType(bClassType);
    var cObject = new ObjectType(cClassType);
    var dObject = new ObjectType(dClassType);
    var intStrUnionObject = new ObjectType(UnionType.or(intClassType, strClassType));
    var intOrStrUnion = UnionType.or(new ObjectType(intClassType), new ObjectType(strClassType));
    var intOrAUnionObject = new ObjectType(UnionType.or(intClassType, aClassType));
    var intOrAUnion = UnionType.or(new ObjectType(intClassType), new ObjectType(aClassType));
    var iObject = new ObjectType(iClassType);
    var jObject = new ObjectType(jClassType);
    var unresolvedObject = new ObjectType(unresolvedType);
    var unresolvedUnionType = new ObjectType(UnionType.or(intClassType, unresolvedType));


    return Stream.of(
      Arguments.of(isIntBuilder, intObjectType, TriBool.TRUE),
      Arguments.of(isIntBuilder, strObjectType, TriBool.FALSE),
      Arguments.of(isIntBuilder, PythonType.UNKNOWN, TriBool.UNKNOWN),
      Arguments.of(isIntBuilder, aObject, TriBool.TRUE),
      Arguments.of(isIntBuilder, bObject, TriBool.FALSE),
      Arguments.of(isIntBuilder, cObject, TriBool.TRUE),
      Arguments.of(isIntBuilder, dObject, TriBool.UNKNOWN),
      Arguments.of(isIntBuilder, intStrUnionObject, TriBool.UNKNOWN),
      Arguments.of(isIntBuilder, intOrStrUnion, TriBool.UNKNOWN),
      Arguments.of(isIntBuilder, intOrAUnionObject, TriBool.TRUE),
      Arguments.of(isIntBuilder, intOrAUnion, TriBool.TRUE),
      Arguments.of(isIntBuilder, iObject, TriBool.TRUE),
      Arguments.of(isIntBuilder, jObject, TriBool.UNKNOWN),
      Arguments.of(isIntBuilder, unresolvedObject, TriBool.UNKNOWN),
      Arguments.of(isIntBuilder, unresolvedUnionType, TriBool.UNKNOWN)
    );
  }

  @ParameterizedTest
  @MethodSource("isInstanceOfTestProvider")
  void isInstanceOfTest(TypeCheckBuilder typeCheckBuilder, PythonType type, TriBool expected) {
    assertThat(typeCheckBuilder.check(type)).isEqualTo(expected);
  }

  @Test
  void unresolvedImportTypeIsSameType() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = spy(new ProjectLevelTypeTable(symbolTable));
    when(table.getType("stubbed.unknown1")).thenReturn(new UnresolvedImportType("stubbed.unknown1"));
    when(table.getType("stubbed.unknown2")).thenReturn(new UnresolvedImportType("stubbed.unknown2"));
    var builder = new TypeCheckBuilder(table).isTypeWithName("stubbed.unknown1");

    var unknown1 = new UnresolvedImportType("stubbed.unknown1");
    var unknown2 = new UnresolvedImportType("stubbed.unknown2");

    assertThat(
      List.of(
        builder.check(unknown1),
        builder.check(unknown2),
        builder.check(PythonType.UNKNOWN)
      )
    ).containsExactly(
      TriBool.TRUE,
      TriBool.UNKNOWN,
      TriBool.UNKNOWN
    );

    var builderUnknownType = new TypeCheckBuilder(table).isTypeWithName("unknown");

    assertThat(
      List.of(
        builderUnknownType.check(unknown1),
        builderUnknownType.check(unknown2),
        builderUnknownType.check(PythonType.UNKNOWN)
      )
    ).containsExactly(
      TriBool.UNKNOWN,
      TriBool.UNKNOWN,
      TriBool.UNKNOWN
    );
  }

  @Test
  void objectTypeThrowsOnDefinitionLocation() {
    var objectTypeBuilder = new ObjectTypeBuilder();
    Assertions.assertThatThrownBy(() -> objectTypeBuilder.withDefinitionLocation(null))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Object type does not have definition location");
  }

  @Test
  void isCompatibleWithTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    BiFunction<PythonType, PythonType, TriBool> isIdentityComparableWith =
      (a, b) -> new TypeCheckBuilder(table).isIdentityComparableWith(a).check(b);

    var intType = table.getType(BuiltinTypes.INT);
    var floatType = table.getType(BuiltinTypes.FLOAT);
    assertThat(isIdentityComparableWith.apply(PythonType.UNKNOWN, PythonType.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isIdentityComparableWith.apply(intType, PythonType.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isIdentityComparableWith.apply(PythonType.UNKNOWN, intType)).isEqualTo(TriBool.UNKNOWN);

    assertThat(isIdentityComparableWith.apply(intType, intType)).isEqualTo(TriBool.TRUE);

    assertThat(isIdentityComparableWith.apply(intType, floatType)).isEqualTo(TriBool.FALSE);
    assertThat(isIdentityComparableWith.apply(floatType, intType)).isEqualTo(TriBool.FALSE);

    var unionType = UnionType.or(intType, floatType);
    assertThat(isIdentityComparableWith.apply(unionType, floatType)).isEqualTo(TriBool.TRUE);
    assertThat(isIdentityComparableWith.apply(floatType, unionType)).isEqualTo(TriBool.TRUE);
    assertThat(isIdentityComparableWith.apply(unionType, unionType)).isEqualTo(TriBool.TRUE);

    assertThat(isIdentityComparableWith.apply(unionType, PythonType.UNKNOWN)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isIdentityComparableWith.apply(PythonType.UNKNOWN, unionType)).isEqualTo(TriBool.UNKNOWN);

    var unknownUnionType = new UnionType(Set.of(PythonType.UNKNOWN, intType));
    assertThat(isIdentityComparableWith.apply(unknownUnionType, intType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isIdentityComparableWith.apply(intType, unknownUnionType)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void isBuiltInWithNameTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    var isIntBuilder = new TypeCheckBuilder(table).isBuiltinOrInstanceWithName("int");

    var intType = table.getType(BuiltinTypes.INT);
    var lazyIntType = table.lazyTypesContext().getOrCreateLazyType(BuiltinTypes.INT);
    var strType = table.getType(BuiltinTypes.STR);
    var lazyStrType = table.lazyTypesContext().getOrCreateLazyType(BuiltinTypes.STR);
    var intStrUnion = UnionType.or(intType, strType);
    var lazyIntStrUnion = new UnionType(Set.of(lazyIntType, strType));
    var unknownType = PythonType.UNKNOWN;

    assertThat(isIntBuilder.check(intType)).isEqualTo(TriBool.TRUE);
    assertThat(isIntBuilder.check(lazyIntType)).isEqualTo(TriBool.TRUE);
    assertThat(isIntBuilder.check(strType)).isEqualTo(TriBool.FALSE);
    assertThat(isIntBuilder.check(lazyStrType)).isEqualTo(TriBool.FALSE);
    assertThat(isIntBuilder.check(intStrUnion)).isEqualTo(TriBool.FALSE);
    assertThat(isIntBuilder.check(lazyIntStrUnion)).isEqualTo(TriBool.FALSE);
    assertThat(isIntBuilder.check(unknownType)).isEqualTo(TriBool.UNKNOWN);
  }


  @Test
  void canBeBuiltInWithNameTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    var canBeIntBuilder = new TypeCheckBuilder(table).canBeBuiltinWithName("int");

    var intType = table.getType(BuiltinTypes.INT);
    var lazyIntType = table.lazyTypesContext().getOrCreateLazyType(BuiltinTypes.INT);
    var strType = table.getType(BuiltinTypes.STR);
    var lazyStrType = table.lazyTypesContext().getOrCreateLazyType(BuiltinTypes.STR);
    var intStrUnion = UnionType.or(intType, strType);
    var lazyIntStrUnion = new UnionType(Set.of(lazyIntType, strType));
    var unknownType = PythonType.UNKNOWN;

    assertThat(canBeIntBuilder.check(intType)).isEqualTo(TriBool.TRUE);
    assertThat(canBeIntBuilder.check(lazyIntType)).isEqualTo(TriBool.TRUE);
    assertThat(canBeIntBuilder.check(strType)).isEqualTo(TriBool.FALSE);
    assertThat(canBeIntBuilder.check(lazyStrType)).isEqualTo(TriBool.FALSE);
    assertThat(canBeIntBuilder.check(intStrUnion)).isEqualTo(TriBool.TRUE);
    assertThat(canBeIntBuilder.check(lazyIntStrUnion)).isEqualTo(TriBool.TRUE);
    assertThat(canBeIntBuilder.check(unknownType)).isEqualTo(TriBool.UNKNOWN);
  }
}
