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
package org.sonar.python.semantic.v2;

import java.util.List;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.TypeCheckBuilder;
import org.sonar.python.types.v2.TypeSource;
import org.sonar.python.types.v2.UnionType;

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
    var table = new ProjectLevelTypeTable(symbolTable, new TypeShed(symbolTable));
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

}
