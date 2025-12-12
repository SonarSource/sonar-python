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
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.VariableDescriptor;
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
}
