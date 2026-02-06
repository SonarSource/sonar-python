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
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.types.v2.SpecialFormType;

import static org.assertj.core.api.Assertions.assertThat;

class VariableDescriptorToPythonTypeConverterTest {
  @Test
  void unsupportedClassTest() {
    var ctx = Mockito.mock(ConversionContext.class);
    var descriptor = Mockito.mock(ClassDescriptor.class);
    var converter = new VariableDescriptorToPythonTypeConverter();
    Assertions.assertThatThrownBy(() -> converter.convert(ctx, descriptor))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessage("Unsupported Descriptor");
  }

  @Test
  void testSpecialFormTypeConversion() {
    var ctx = Mockito.mock(ConversionContext.class);
    var converter = new VariableDescriptorToPythonTypeConverter();

    var typingSelfDescriptor = new VariableDescriptor("Self", "typing.Self", "typing._SpecialForm");
    var typingExtensionsSelfDescriptor = new VariableDescriptor("Self", "typing_extensions.Self", "typing_extensions._SpecialForm");

    Assertions.assertThat(converter.convert(ctx, typingSelfDescriptor))
      .isInstanceOfSatisfying(SpecialFormType.class,
        specialFormType -> assertThat(specialFormType.fullyQualifiedName()).isEqualTo("typing.Self"));

    Assertions.assertThat(converter.convert(ctx, typingExtensionsSelfDescriptor))
      .isInstanceOfSatisfying(SpecialFormType.class,
        specialFormType -> assertThat(specialFormType.fullyQualifiedName()).isEqualTo("typing_extensions.Self"));
  }

  @Test
  void testConvertWithAttributes() {
    var lazyTypesContext = new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    var converter = new VariableDescriptorToPythonTypeConverter();

    VariableDescriptor intDesc = new VariableDescriptor("int", "builtins.int", "int");
    VariableDescriptor listDesc = new VariableDescriptor(
      "x", "mod.x", "builtins.list", false, List.of(intDesc), List.of()
    );

    DescriptorToPythonTypeConverter delegateConverter = (ctx, desc) -> {
      if (desc instanceof VariableDescriptor varDesc) {
        return converter.convert(ctx, varDesc);
      }
      return PythonType.UNKNOWN;
    };
    var ctx = new ConversionContext("mod", lazyTypesContext, delegateConverter, null);

    PythonType result = converter.convert(ctx, listDesc);

    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType objType = (ObjectType) result;
    assertThat(objType.attributes()).hasSize(1);
  }

  @Test
  void testConvertWithMembers() {
    var lazyTypesContext = new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    var converter = new VariableDescriptorToPythonTypeConverter();

    VariableDescriptor memberDesc = new VariableDescriptor("value", "mod.x.value", "str");
    VariableDescriptor varDesc = new VariableDescriptor(
      "x", "mod.x", "SomeClass", false, List.of(), List.of(memberDesc)
    );

    DescriptorToPythonTypeConverter delegateConverter = (ctx, desc) -> {
      if (desc instanceof VariableDescriptor vd) {
        return converter.convert(ctx, vd);
      }
      return PythonType.UNKNOWN;
    };
    var ctx = new ConversionContext("mod", lazyTypesContext, delegateConverter, null);

    PythonType result = converter.convert(ctx, varDesc);

    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType objType = (ObjectType) result;
    assertThat(objType.members()).hasSize(1);
    assertThat(objType.members().get(0).name()).isEqualTo("value");
  }

  @Test
  void testConvertWithAttributesAndMembers() {
    var lazyTypesContext = new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    var converter = new VariableDescriptorToPythonTypeConverter();

    VariableDescriptor intDesc = new VariableDescriptor("int", "builtins.int", "int");
    VariableDescriptor memberDesc = new VariableDescriptor("value", "mod.x.value", "str");
    VariableDescriptor varDesc = new VariableDescriptor(
      "x", "mod.x", "SomeGeneric", false, List.of(intDesc), List.of(memberDesc)
    );

    DescriptorToPythonTypeConverter delegateConverter = (ctx, desc) -> {
      if (desc instanceof VariableDescriptor vd) {
        return converter.convert(ctx, vd);
      }
      return PythonType.UNKNOWN;
    };
    var ctx = new ConversionContext("mod", lazyTypesContext, delegateConverter, null);

    PythonType result = converter.convert(ctx, varDesc);

    assertThat(result).isInstanceOf(ObjectType.class);
    ObjectType objType = (ObjectType) result;
    assertThat(objType.attributes()).hasSize(1);
    assertThat(objType.members()).hasSize(1);
    assertThat(objType.members().get(0).name()).isEqualTo("value");
  }
}
