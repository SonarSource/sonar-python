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
package org.sonar.python.semantic.v2;

import java.util.List;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType.UnresolvedImportType;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.types.v2.TypeCheckBuilder;

import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

class TypeCheckerBuilderTest {

  @Test
  void typeSourceTest() {
    var builder = new TypeCheckBuilder(null).isTypeHintTypeSource();
    Assertions.assertThat(builder.check(ObjectType.Builder.fromType(PythonType.UNKNOWN).withTypeSource(TypeSource.TYPE_HINT).build()))
      .isEqualTo(TriBool.TRUE);
    Assertions.assertThat(builder.check(ObjectType.fromType(PythonType.UNKNOWN)))
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

    var intObjectType = ObjectType.fromType(intClassType);
    var strObjectType = ObjectType.fromType(strClassType);
    var aObject = ObjectType.fromType(aClassType);
    var bObject = ObjectType.fromType(bClassType);
    var cObject = ObjectType.fromType(cClassType);
    var dObject = ObjectType.fromType(dClassType);
    var eObject = ObjectType.fromType(UnionType.or(intClassType, strClassType));
    var fObject = UnionType.or(ObjectType.fromType(intClassType), ObjectType.fromType(strClassType));
    var gObject = ObjectType.fromType(UnionType.or(intClassType, aClassType));
    var hObject = UnionType.or(ObjectType.fromType(intClassType), ObjectType.fromType(aClassType));
    var iObject = ObjectType.fromType(iClassType);
    var jObject = ObjectType.fromType(jClassType);
    var unresolvedObject = ObjectType.fromType(unresolvedType);
    var unresolvedUnionObject = ObjectType.fromType(UnionType.or(intClassType, unresolvedType));

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
    var fooObjectType = ObjectType.fromType(fooClassType);
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
    var objectTypeBuilder = ObjectType.Builder.fromType(PythonType.UNKNOWN);
    Assertions.assertThatThrownBy(() -> objectTypeBuilder.withDefinitionLocation(null))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("Object type does not have definition location");
  }

  @Test
  void selfTypeHandlingTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    var intType = table.getType("int");
    var selfType = (SelfType) SelfType.of(intType);
    
    Assertions.assertThat(new TypeCheckBuilder(table).isTypeOrInstanceWithName("int").check(selfType))
      .isEqualTo(TriBool.UNKNOWN);

    Assertions.assertThat(new TypeCheckBuilder(table).isInstanceOf("int").check(selfType))
      .isEqualTo(TriBool.UNKNOWN);
    
    var objectIntType = ObjectType.fromType(intType);
    var selfWrappedInObject = SelfType.of(objectIntType);
    Assertions.assertThat(new TypeCheckBuilder(table).isInstance().check(selfWrappedInObject))
      .isEqualTo(TriBool.TRUE);
    
    Assertions.assertThat(new TypeCheckBuilder(table).isInstance().check(selfType))
      .isEqualTo(TriBool.FALSE);
    
    var mockTable = spy(table);
    when(mockTable.getType("int")).thenReturn(selfType);
    Assertions.assertThat(new TypeCheckBuilder(mockTable).isTypeOrInstanceWithName("int").check(intType))
      .isEqualTo(TriBool.UNKNOWN);
    Assertions.assertThat(new TypeCheckBuilder(mockTable).isInstanceOf("int").check(objectIntType))
      .isEqualTo(TriBool.UNKNOWN);
  }

}
