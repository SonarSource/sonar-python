/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.semantic.v2;

import java.util.List;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType.UnresolvedImportType;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.v2.TypeCheckBuilder;

import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

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

    var aClassType = new ClassTypeBuilder("A", "mod.A")
      .withSuperClasses(intClassType)
      .build();

    var bClassType = new ClassTypeBuilder("B", "mod.B")
      .withSuperClasses(strClassType)
      .build();

    var cClassType = new ClassTypeBuilder("C", "mod.C")
      .withSuperClasses(PythonType.UNKNOWN, intClassType)
      .build();

    var dClassType = new ClassTypeBuilder("D", "mod.D")
      .withSuperClasses(PythonType.UNKNOWN, strClassType)
      .build();

    var iClassType = new ClassTypeBuilder("I", "mod.I")
      .withSuperClasses(aClassType, intClassType)
      .build();

    var jClassType = new ClassTypeBuilder("J", "mod.J")
      .withSuperClasses(UnionType.or(bClassType, aClassType))
      .build();

    var unresolvedType = new UnresolvedImportType("unknown");

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
    var unresolvedObject = new ObjectType(unresolvedType);
    var unresolvedUnionObject = new ObjectType(UnionType.or(intClassType, unresolvedType));

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
        builder.check(jObject),
        builder.check(unresolvedObject),
        builder.check(unresolvedUnionObject)
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
      TriBool.UNKNOWN,
      TriBool.UNKNOWN,
      TriBool.UNKNOWN
    );
  }

  @Test
  void isTypeWithFqnTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    var intBuilder = new TypeCheckBuilder(table).isTypeWithFqn("int");
    var fooBuilder = new TypeCheckBuilder(table).isTypeWithFqn("mod.foo");

    var intClassType = table.getType("int");
    var strClassType = table.getType("str");
    var fooClassType = new ClassTypeBuilder("foo", "mod.foo").build();
    var fooFunctionType = new FunctionTypeBuilder("foo").withFullyQualifiedName("mod.foo").build();
    var fooUnresolvedImportType = new UnresolvedImportType("mod.foo");
    var fooObjectType = new ObjectType(fooClassType);
    var fooUnionType = UnionType.or(fooClassType, fooFunctionType);
    var barClassType = new ClassTypeBuilder("bar", "mod.bar").build();

    Assertions.assertThat(
      List.of(
        intBuilder.check(intClassType),
        intBuilder.check(strClassType),
        fooBuilder.check(fooClassType),
        fooBuilder.check(fooFunctionType),
        fooBuilder.check(fooUnresolvedImportType),
        fooBuilder.check(fooObjectType),
        fooBuilder.check(fooUnionType),
        fooBuilder.check(barClassType),
        fooBuilder.check(PythonType.UNKNOWN)
      )
    ).containsExactly(
      TriBool.TRUE,
      TriBool.FALSE,
      TriBool.TRUE,
      TriBool.TRUE,
      TriBool.TRUE,
      TriBool.UNKNOWN,
      TriBool.UNKNOWN,
      TriBool.FALSE,
      TriBool.UNKNOWN
    );
  }

  @Test
  void unresolvedImportTypeIsSameType() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = spy(new ProjectLevelTypeTable(symbolTable));
    when(table.getType("stubbed.unknown1")).thenReturn(new UnresolvedImportType("stubbed.unknown1"));
    when(table.getType("stubbed.unknown2")).thenReturn(new UnresolvedImportType("stubbed.unknown2"));
    var builder = new TypeCheckBuilder(table).isTypeOrInstanceWithName("stubbed.unknown1");

    var unknown1 = new UnresolvedImportType("stubbed.unknown1");
    var unknown2 = new UnresolvedImportType("stubbed.unknown2");

    Assertions.assertThat(
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

    var builderUnknownType = new TypeCheckBuilder(table).isTypeOrInstanceWithName("unknown");

    Assertions.assertThat(
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

}
