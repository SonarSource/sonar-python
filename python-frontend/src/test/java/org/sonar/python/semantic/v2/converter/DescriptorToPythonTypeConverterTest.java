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
package org.sonar.python.semantic.v2.converter;

import java.util.List;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeWrapper;

class DescriptorToPythonTypeConverterTest {

  @Test
  void unknownDescriptorToPythonTypeConverterTest() {
    var ctx = Mockito.mock(ConversionContext.class);
    var descriptor = Mockito.mock(Descriptor.class);
    Mockito.when(descriptor.kind()).thenReturn(null);
    var converter = new UnknownDescriptorToPythonTypeConverter();
    var type = converter.convert(ctx, descriptor);
    Assertions.assertThat(type).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void ambiguousDescriptorConversionTest() {
    var lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    var converter = new AnyDescriptorToPythonTypeConverter(lazyTypesContext);
    var descriptor = Mockito.mock(AmbiguousDescriptor.class);
    Mockito.when(descriptor.kind()).thenReturn(Descriptor.Kind.AMBIGUOUS);

    var type = converter.convert(descriptor);
    Assertions.assertThat(type).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void classDescriptorConversionTest() {
    var lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    var converter = new AnyDescriptorToPythonTypeConverter(lazyTypesContext);
    var descriptor = Mockito.mock(ClassDescriptor.class);

    var parentClassName = "Parent";
    var resolvedParent = new ClassTypeBuilder().withName(parentClassName).build();

    var member = Mockito.mock(AmbiguousDescriptor.class);
    Mockito.when(member.kind()).thenReturn(Descriptor.Kind.AMBIGUOUS);
    Mockito.when(member.name()).thenReturn("member");

    Mockito.when(descriptor.kind()).thenReturn(Descriptor.Kind.CLASS);
    Mockito.when(descriptor.name()).thenReturn("Sample");
    Mockito.when(descriptor.superClasses()).thenReturn(List.of(parentClassName));
    Mockito.when(descriptor.members()).thenReturn(List.of(
      member
    ));
    Mockito.when(lazyTypesContext.getOrCreateLazyType(parentClassName))
      .thenReturn(new LazyType(parentClassName, lazyTypesContext));

    Mockito.when(lazyTypesContext.resolveLazyType(Mockito.argThat(lt -> parentClassName.equals(lt.fullyQualifiedName()))))
      .thenReturn(resolvedParent);

    var type = (ClassType) converter.convert(descriptor);
    Assertions.assertThat(type.name()).isEqualTo("Sample");
    
    Assertions.assertThat(type.superClasses())
      .extracting(TypeWrapper.class::cast)
      .extracting(TypeWrapper::type)
      .containsOnly(resolvedParent);

    Assertions.assertThat(type.members())
      .extracting(Member::name)
      .containsOnly("member");
  }



}
