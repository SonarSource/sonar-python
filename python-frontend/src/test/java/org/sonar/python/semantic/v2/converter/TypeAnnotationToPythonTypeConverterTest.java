/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.semantic.v2.converter;

import java.util.List;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class TypeAnnotationToPythonTypeConverterTest {

  @ParameterizedTest
  @CsvSource({"TUPLE, tuple, FALSE", "TYPE_VAR, int, FALSE", "TYPED_DICT, dict, FALSE", "TYPE, type, TRUE", "CALLABLE, function, TRUE"})
  void genericArgumentsDescribeRuntimeValues(TypeAnnotationDescriptor.TypeKind kind, String fqn, TriBool callability) {
    var ctx = mock(ConversionContext.class);
    when(ctx.lazyTypesContext()).thenReturn(new LazyTypesContext(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE));
    var argument = new TypeAnnotationDescriptor(fqn, kind, List.of(), fqn, false);
    var annotation = new TypeAnnotationDescriptor("list", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(argument), "list", false);

    var result = (ObjectType) new TypeAnnotationToPythonTypeConverter().convert(ctx, annotation);

    assertThat(result.attributes()).singleElement().satisfies(value -> {
      assertThat(value).is(TypesTestUtils.objectTypeOf(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE.getType(fqn)));
      assertThat(value.hasMember("__call__")).isEqualTo(callability);
    });
  }

  @Test
  void selfTypeVariableDescribesAnInstanceWithoutWrappingAnObjectInSelf() {
    var ctx = mock(ConversionContext.class);
    when(ctx.lazyTypesContext()).thenReturn(new LazyTypesContext(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE));
    var bound = new TypeAnnotationDescriptor("list", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "list", false);
    var self = new TypeAnnotationDescriptor("Self", TypeAnnotationDescriptor.TypeKind.TYPE_VAR, List.of(bound), "list", true);

    var result = new TypeAnnotationToPythonTypeConverter().convert(ctx, self);

    assertThat(result).isInstanceOfSatisfying(ObjectType.class, value ->
      assertThat(value.unwrappedType()).isInstanceOfSatisfying(SelfType.class, selfType ->
        assertThat(selfType.innerType()).isEqualTo(TypesTestUtils.LIST_TYPE)));
  }

  @Test
  void noneAnnotationsDescribeValuesRatherThanCallableClasses() {
    var ctx = mock(ConversionContext.class);
    when(ctx.lazyTypesContext()).thenReturn(new LazyTypesContext(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE));
    var none = new TypeAnnotationDescriptor("None", TypeAnnotationDescriptor.TypeKind.NONE, List.of(), "NoneType", false);
    var converter = new TypeAnnotationToPythonTypeConverter();

    var result = converter.convert(ctx, none);
    assertThat(result).isInstanceOf(ObjectType.class);
    assertThat(result.hasMember("__call__")).isEqualTo(org.sonar.plugins.python.api.TriBool.FALSE);

    // Current Typeshed declares OSError.errno as int | None. Both alternatives are non-callable values.
    var integer = new TypeAnnotationDescriptor("int", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "int", false);
    var optionalInteger = new TypeAnnotationDescriptor("int | None", TypeAnnotationDescriptor.TypeKind.UNION, List.of(integer, none), null, false);
    var union = (org.sonar.python.types.v2.ResolvableType) converter.convert(ctx, optionalInteger);
    assertThat(union.resolve().hasMember("__call__"))
      .isEqualTo(org.sonar.plugins.python.api.TriBool.FALSE);
  }

  @Test
  void typeVarWithIncorrectArgsAndIsSelfTrueReturnsUnknown() {
    var lazyTypesContext = mock(LazyTypesContext.class);
    var ctx = mock(ConversionContext.class);
    var converter = new TypeAnnotationToPythonTypeConverter();

    when(ctx.lazyTypesContext()).thenReturn(lazyTypesContext);

    // Create a TYPE_VAR TypeAnnotationDescriptor with isSelf=true and multiple args
    var innerArg1 = new TypeAnnotationDescriptor("int", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "int", false);
    var innerArg2 = new TypeAnnotationDescriptor("str", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "str", false);
    var typeVarDescriptor = new TypeAnnotationDescriptor(
      "TypeVar",
      TypeAnnotationDescriptor.TypeKind.TYPE_VAR,
      List.of(innerArg1, innerArg2),
      "typing.TypeVar",
      true
    );

    var result = converter.convert(ctx, typeVarDescriptor);

    Assertions.assertThat(result)
      .as("TYPE_VAR with isSelf=true and args.size() > 1 should return UNKNOWN")
      .isEqualTo(PythonType.UNKNOWN);


    // Create a TYPE_VAR TypeAnnotationDescriptor with isSelf=true and zero args
     typeVarDescriptor = new TypeAnnotationDescriptor(
      "TypeVar",
      TypeAnnotationDescriptor.TypeKind.TYPE_VAR,
      List.of(),
      "typing.TypeVar",
      true
    );
     result = converter.convert(ctx, typeVarDescriptor);

     Assertions.assertThat(result)
       .as("TYPE_VAR with isSelf=true and args.size() == 0 should return UNKNOWN")
       .isEqualTo(PythonType.UNKNOWN);
   }

   @Test
   void testListInstanceWithAttributes() {
     var lazyTypesContext = new LazyTypesContext(TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE);
     var ctx = mock(ConversionContext.class);
     var converter = new TypeAnnotationToPythonTypeConverter();

     when(ctx.lazyTypesContext()).thenReturn(lazyTypesContext);

     // Create a INSTANCE TypeAnnotationDescriptor for list[int]
     var intArg = new TypeAnnotationDescriptor("int", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "int", false);
     var typeVarDescriptor = new TypeAnnotationDescriptor(
       "list",
       TypeAnnotationDescriptor.TypeKind.INSTANCE,
       List.of(intArg),
       "builtins.list",
       true);

     var result = converter.convert(ctx, typeVarDescriptor);

     assertThat(result).isInstanceOf(ObjectType.class);
     ObjectType objType = (ObjectType) result;
     assertThat(objType.attributes()).hasSize(1);
     assertThat(objType.attributes().get(0)).is(TypesTestUtils.objectTypeOf(TypesTestUtils.INT_TYPE));
  }
}
