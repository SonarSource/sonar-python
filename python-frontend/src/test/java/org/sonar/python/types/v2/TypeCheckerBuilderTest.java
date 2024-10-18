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
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.ObjectTypeBuilder;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;

import static org.fest.assertions.Assertions.assertThat;

class TypeCheckerBuilderTest {

  @Test
  void typeSourceTest() {
    var builder = new TypeCheckBuilder(null).isTypeHintTypeSource();
    Assertions.assertThat(builder.check(new ObjectType(PythonType.UNKNOWN, TypeSource.TYPE_HINT)))
      .isEqualTo(TriBool.TRUE);
    Assertions.assertThat(builder.check(new ObjectType(PythonType.UNKNOWN, TypeSource.EXACT)))
      .isEqualTo(TriBool.FALSE);
  }

  @Test
  void isInstanceOfTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    var builder = new TypeCheckBuilder(table).isInstanceOf("int");

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

    var intObjectType = new ObjectType(intClassType);
    var strObjectType = new ObjectType(strClassType);
    var aObject = new ObjectType(aClassType);
    var bObject = new ObjectType(bClassType);
    var cObject = new ObjectType(cClassType);
    var dObject = new ObjectType(dClassType);
    var eObject = new ObjectType(UnionType.or(intClassType, strClassType));
    var fObject = UnionType.or(new ObjectType(intClassType), new ObjectType(strClassType));
    var gObject = new ObjectType(UnionType.or(intClassType, aClassType));
    var hObject = UnionType.or(new ObjectType(intClassType), new ObjectType(aClassType));
    var iObject = new ObjectType(iClassType);
    var jObject = new ObjectType(jClassType);

    Assertions.assertThat(
      List.of(
        builder.check(intObjectType),
        builder.check(strObjectType),
        builder.check(PythonType.UNKNOWN),
        builder.check(aObject),
        builder.check(bObject),
        builder.check(cObject),
        builder.check(dObject),
        builder.check(eObject),
        builder.check(fObject),
        builder.check(gObject),
        builder.check(hObject),
        builder.check(iObject),
        builder.check(jObject)
      )
    ).containsExactly(
      TriBool.TRUE,
      TriBool.FALSE,
      TriBool.UNKNOWN,
      TriBool.TRUE,
      TriBool.FALSE,
      TriBool.TRUE,
      TriBool.UNKNOWN,
      TriBool.UNKNOWN,
      TriBool.UNKNOWN,
      TriBool.TRUE,
      TriBool.TRUE,
      TriBool.TRUE,
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
    BiFunction<PythonType, PythonType, TriBool> isIdentityComparableWith = (a, b) -> new TypeCheckBuilder(table).isIdentityComparableWith(a).check(b);

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
}
