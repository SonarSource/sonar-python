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
package org.sonar.python.semantic.v2.converter;

import java.util.List;
import java.util.Set;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.plugins.python.api.types.v2.Member;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;

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

    var descriptorAlternative1 = Mockito.mock(FunctionDescriptor.class);

    var returnTypeName = "Returned";
    String fullyQualifiedName = "returned.ReturnTypeName";
    var resolvedReturnType = new ClassTypeBuilder(returnTypeName, fullyQualifiedName).build();
    Mockito.when(lazyTypesContext.resolveLazyType(Mockito.argThat(lt -> returnTypeName.equals(lt.importPath()))))
      .thenReturn(resolvedReturnType);

    Mockito.when(descriptorAlternative1.kind()).thenReturn(Descriptor.Kind.FUNCTION);
    Mockito.when(descriptorAlternative1.name()).thenReturn("Sample");
    Mockito.when(descriptorAlternative1.annotatedReturnTypeName()).thenReturn(returnTypeName);
    Mockito.when(descriptorAlternative1.parameters()).thenReturn(List.of(
      new FunctionDescriptor.Parameter(
        "p1",
        "Returned",
        false,
        false,
        true,
        false,
        false,
        new LocationInFile("m1", 1, 10,  1, 15))
    ));

    var descriptorAlternative2 = Mockito.mock(FunctionDescriptor.class);
    Mockito.when(descriptorAlternative2.kind()).thenReturn(Descriptor.Kind.FUNCTION);
    Mockito.when(descriptorAlternative2.name()).thenReturn("Sample");
    Mockito.when(descriptorAlternative2.annotatedReturnTypeName()).thenReturn(returnTypeName);
    Mockito.when(descriptorAlternative2.parameters()).thenReturn(List.of(
      new FunctionDescriptor.Parameter(
        "p2",
        "Returned",
        false,
        false,
        true,
        false,
        false,
        new LocationInFile("m1", 2, 10,  2, 15))
    ));

    var descriptor = Mockito.mock(AmbiguousDescriptor.class);
    Mockito.when(descriptor.kind()).thenReturn(Descriptor.Kind.AMBIGUOUS);
    Mockito.when(descriptor.alternatives()).thenReturn(Set.of(descriptorAlternative1, descriptorAlternative2));

    var type = (UnionType) converter.convert("", descriptor, TypeOrigin.LOCAL);
    Assertions.assertThat(type.candidates())
      .hasSize(2);

    Assertions.assertThat(type.candidates())
      .extracting(FunctionType.class::cast)
      .flatExtracting(FunctionType::parameters)
      .extracting(ParameterV2::name)
      .containsOnly("p1", "p2");

  }

  @Test
  void classDescriptorConversionTest() {
    var lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    var converter = new AnyDescriptorToPythonTypeConverter(lazyTypesContext);
    var descriptor = Mockito.mock(ClassDescriptor.class);

    var parentClassName = "Parent";
    String parentClassFqn = "parent.Parent";
    var resolvedParent = new ClassTypeBuilder(parentClassName, parentClassFqn).build();

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

    Mockito.when(lazyTypesContext.resolveLazyType(Mockito.argThat(lt -> parentClassName.equals(lt.importPath()))))
      .thenReturn(resolvedParent);

    var type = (ClassType) converter.convert("", descriptor, TypeOrigin.LOCAL);
    Assertions.assertThat(type.name()).isEqualTo("Sample");
    
    Assertions.assertThat(type.superClasses())
      .extracting(TypeWrapper.class::cast)
      .extracting(TypeWrapper::type)
      .containsOnly(resolvedParent);

    Assertions.assertThat(type.members())
      .extracting(Member::name)
      .containsOnly("member");
  }

  @Test
  void functionDescriptorConversionTest() {
    var lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    var converter = new AnyDescriptorToPythonTypeConverter(lazyTypesContext);
    var descriptor = Mockito.mock(FunctionDescriptor.class);

    var returnTypeName = "Returned";
    String returnTypeFqn = "returned.Returned";
    var resolvedReturnType = new ClassTypeBuilder(returnTypeName, returnTypeFqn).build();

    Mockito.when(descriptor.kind()).thenReturn(Descriptor.Kind.FUNCTION);
    Mockito.when(descriptor.name()).thenReturn("Sample");
    Mockito.when(descriptor.annotatedReturnTypeName()).thenReturn(returnTypeName);

    TypeAnnotationDescriptor typeAnnotationDescriptor = Mockito.mock(TypeAnnotationDescriptor.class);
    Mockito.when(typeAnnotationDescriptor.kind()).thenReturn(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    Mockito.when(typeAnnotationDescriptor.fullyQualifiedName()).thenReturn(returnTypeName);
    Mockito.when(descriptor.typeAnnotationDescriptor()).thenReturn(typeAnnotationDescriptor);

    Mockito.when(descriptor.parameters()).thenReturn(List.of(
      new FunctionDescriptor.Parameter(
        "p1",
        "Returned",
        false,
        false,
        true,
        false,
        false,
        new LocationInFile("m1", 1, 10,  1, 15))
    ));

    Mockito.when(lazyTypesContext.getOrCreateLazyType(returnTypeName))
      .thenReturn(new LazyType(returnTypeName, lazyTypesContext));

    Mockito.when(lazyTypesContext.resolveLazyType(Mockito.argThat(lt -> returnTypeName.equals(lt.importPath()))))
      .thenReturn(resolvedReturnType);

    var type = (FunctionType) converter.convert("", descriptor, TypeOrigin.LOCAL);
    Assertions.assertThat(type.name()).isEqualTo("Sample");

    Assertions.assertThat(type.parameters()).hasSize(1);
    var parameter = type.parameters().get(0);
    Assertions.assertThat(parameter.declaredType())
      .extracting(TypeWrapper::type)
      .extracting(PythonType::unwrappedType)
      .isSameAs(resolvedReturnType);
    Assertions.assertThat(type.returnType())
      .extracting(PythonType::unwrappedType)
      .isSameAs(resolvedReturnType);
  }

  @Test
  void variableDescriptorConversionTest() {
    var lazyTypesContext = Mockito.mock(LazyTypesContext.class);
    var converter = new AnyDescriptorToPythonTypeConverter(lazyTypesContext);
    var descriptor = Mockito.mock(VariableDescriptor.class);

    var variableTypeName = "Returned";
    String variableTypeFqn = "returned.Returned";
    var resolvedVariableType = new ClassTypeBuilder(variableTypeName, variableTypeFqn).build();

    Mockito.when(descriptor.kind()).thenReturn(Descriptor.Kind.VARIABLE);
    Mockito.when(descriptor.name()).thenReturn("variable");
    Mockito.when(descriptor.annotatedType()).thenReturn(variableTypeName);

    Mockito.when(lazyTypesContext.getOrCreateLazyTypeWrapper(variableTypeName))
      .thenReturn(new LazyTypeWrapper(new LazyType(variableTypeName, lazyTypesContext)));

    Mockito.when(lazyTypesContext.resolveLazyType(Mockito.argThat(lt -> variableTypeName.equals(lt.importPath()))))
      .thenReturn(resolvedVariableType);

    var type = (ObjectType) converter.convert("", descriptor, TypeOrigin.LOCAL);
    Assertions.assertThat(type)
      .extracting(PythonType::unwrappedType)
      .isSameAs(resolvedVariableType);
  }

}
