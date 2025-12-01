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

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;

import static org.assertj.core.api.Assertions.assertThat;

class ClassDescriptorToPythonTypeConverterTest {
  @Test
  void unsupportedClassTest() {
    var ctx = Mockito.mock(ConversionContext.class);
    var descriptor = Mockito.mock(FunctionDescriptor.class);
    var converter = new ClassDescriptorToPythonTypeConverter();
    Assertions.assertThatThrownBy(() -> converter.convert(ctx, descriptor))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessage("Unsupported Descriptor");
  }

  @Test
  void classDescriptorWithIsSelfConvertedToSelfType() {
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("MyClass")
      .withFullyQualifiedName("mod.MyClass")
      .withIsSelf(true)
      .build();

    var lazyTypesContext = new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    var ctx = new ConversionContext("mod", lazyTypesContext, (c, d) -> PythonType.UNKNOWN, TypeOrigin.STUB);
    var converter = new ClassDescriptorToPythonTypeConverter();

    PythonType result = converter.convert(ctx, classDescriptor);

    assertThat(result).isInstanceOf(SelfType.class);
    SelfType selfType = (SelfType) result;
    assertThat(selfType.innerType()).isInstanceOf(ClassType.class);
    ClassType classType = (ClassType) selfType.innerType();
    assertThat(classType.name()).isEqualTo("MyClass");
    assertThat(classType.fullyQualifiedName()).isEqualTo("mod.MyClass");
  }

  @Test
  void classDescriptorWithoutIsSelfConvertedToClassType() {
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("MyClass")
      .withFullyQualifiedName("mod.MyClass")
      .build();

    var lazyTypesContext = new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    var ctx = new ConversionContext("mod", lazyTypesContext, (c, d) -> PythonType.UNKNOWN, TypeOrigin.STUB);
    var converter = new ClassDescriptorToPythonTypeConverter();

    PythonType result = converter.convert(ctx, classDescriptor);

    assertThat(result).isInstanceOf(ClassType.class);
    ClassType classType = (ClassType) result;
    assertThat(classType.name()).isEqualTo("MyClass");
    assertThat(classType.fullyQualifiedName()).isEqualTo("mod.MyClass");
  }
}
