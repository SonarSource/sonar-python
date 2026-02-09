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
package org.sonar.python.semantic.v2.converter;

import java.util.List;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;

class TypeAnnotationToPythonTypeConverterTest {

  @Test
  void typeVarWithIncorrectArgsAndIsSelfTrueReturnsUnknown() {
    var lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    var ctx = Mockito.mock(ConversionContext.class);
    var converter = new TypeAnnotationToPythonTypeConverter();

    Mockito.when(ctx.lazyTypesContext()).thenReturn(lazyTypesContext);

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
     var ctx = Mockito.mock(ConversionContext.class);
     var converter = new TypeAnnotationToPythonTypeConverter();

     Mockito.when(ctx.lazyTypesContext()).thenReturn(lazyTypesContext);

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
