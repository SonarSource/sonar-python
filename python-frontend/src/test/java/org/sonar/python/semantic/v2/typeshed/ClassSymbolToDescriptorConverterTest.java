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
package org.sonar.python.semantic.v2.typeshed;

import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.types.protobuf.SymbolsProtos;

class ClassSymbolToDescriptorConverterTest {

  @Test
  void test() {
    var functionConverter = new FunctionSymbolToDescriptorConverter();
    var variableConverter = new VarSymbolToDescriptorConverter();
    var overloadedFunctionConverter = new OverloadedFunctionSymbolToDescriptorConverter(functionConverter);
    var converter = new ClassSymbolToDescriptorConverter(variableConverter, functionConverter, overloadedFunctionConverter, ProjectPythonVersion.currentVersionValues());

    var symbol = SymbolsProtos.ClassSymbol.newBuilder()
      .setName("MyClass")
      .setFullyQualifiedName("module.MyClass")
      .addSuperClasses("module.AnotherClass")
      .setMetaclassName("module.MetaClass")
      .setHasMetaclass(true)
      .setHasDecorators(true)
      .addAttributes(SymbolsProtos.VarSymbol.newBuilder()
        .setName("v1")
        .setFullyQualifiedName("module.MyClass.v1")
        .build())
      .addMethods(SymbolsProtos.FunctionSymbol.newBuilder()
        .setName("foo")
        .setFullyQualifiedName("module.MyClass.foo")
        .build())
      .addOverloadedMethods(SymbolsProtos.OverloadedFunctionSymbol.newBuilder()
        .setName("overloaded_foo")
        .setFullname("module.MyClass.overloaded_foo")
        .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
          .setName("overloaded_foo")
          .setFullyQualifiedName("module.MyClass.overloaded_foo")
          .build())
        .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
          .setName("overloaded_foo")
          .setFullyQualifiedName("module.MyClass.overloaded_foo")
          .build())
        .build())
      .build();

    var descriptor = converter.convert(symbol);

    Assertions.assertThat(descriptor.name()).isEqualTo("MyClass");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("module.MyClass");
    Assertions.assertThat(descriptor.superClasses()).hasSize(1).containsOnly("module.AnotherClass");
    Assertions.assertThat(descriptor.metaclassFQN()).isEqualTo("module.MetaClass");
    Assertions.assertThat(descriptor.hasMetaClass()).isTrue();
    Assertions.assertThat(descriptor.hasDecorators()).isTrue();
    Assertions.assertThat(descriptor.members()).hasSize(3);

    var membersByName = descriptor.members()
      .stream()
      .collect(Collectors.toMap(Descriptor::name, Function.identity()));

    Assertions.assertThat(membersByName).hasSize(3);
    Assertions.assertThat(membersByName.get("v1")).isInstanceOf(VariableDescriptor.class);
    Assertions.assertThat(membersByName.get("foo")).isInstanceOf(FunctionDescriptor.class);
    Assertions.assertThat(membersByName.get("overloaded_foo")).isInstanceOf(AmbiguousDescriptor.class);

    var foo = (FunctionDescriptor) membersByName.get("foo");
    var overloadedFoo = (AmbiguousDescriptor) membersByName.get("overloaded_foo");
    var overloadedFooCandidates = overloadedFoo.alternatives().stream().toList();

    Assertions.assertThat(foo.isInstanceMethod()).isTrue();
    Assertions.assertThat(overloadedFooCandidates)
      .extracting(FunctionDescriptor.class::cast)
      .extracting(FunctionDescriptor::isInstanceMethod)
      .containsOnly(true, true);
  }

  static Stream<Arguments> versionsToTest() {
    return Stream.of(
      Arguments.of(PythonVersionUtils.Version.V_312),
      Arguments.of(PythonVersionUtils.Version.V_313)
    );
  }

  @ParameterizedTest
  @MethodSource("versionsToTest")
  void validForPythonVersionsTest(PythonVersionUtils.Version version) {
    var functionConverter = new FunctionSymbolToDescriptorConverter();
    var variableConverter = new VarSymbolToDescriptorConverter();
    var overloadedFunctionConverter = new OverloadedFunctionSymbolToDescriptorConverter(functionConverter);
    var converter = new ClassSymbolToDescriptorConverter(variableConverter, functionConverter, overloadedFunctionConverter, Set.of(version.serializedValue()));

    var symbol = SymbolsProtos.ClassSymbol.newBuilder()
      .setName("MyClass")
      .addAttributes(SymbolsProtos.VarSymbol.newBuilder()
        .setName("v1")
        .addValidFor("311")
        .build())
      .addAttributes(SymbolsProtos.VarSymbol.newBuilder()
        .setName("v2")
        .addValidFor("39")
        .build())
      .addMethods(SymbolsProtos.FunctionSymbol.newBuilder()
        .setName("foo1")
        .addValidFor("311")
        .build())
      .addMethods(SymbolsProtos.FunctionSymbol.newBuilder()
        .setName("foo2")
        .addValidFor("39")
        .build())
      .addOverloadedMethods(SymbolsProtos.OverloadedFunctionSymbol.newBuilder()
        .setName("overloaded_foo1")
        .addValidFor("311")
        .setFullname("module.MyClass.overloaded_foo1")
        .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
          .setName("overloaded_foo1")
          .addValidFor("311")
          .build())
        .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
          .setName("overloaded_foo1")
          .addValidFor("311")
          .build())
        .build())
      .addOverloadedMethods(SymbolsProtos.OverloadedFunctionSymbol.newBuilder()
        .setName("overloaded_foo2")
        .addValidFor("39")
        .setFullname("module.MyClass.overloaded_foo2")
        .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
          .setName("overloaded_foo2")
          .addValidFor("39")
          .build())
        .addDefinitions(SymbolsProtos.FunctionSymbol.newBuilder()
          .setName("overloaded_foo2")
          .addValidFor("39")
          .build())
        .build())
      .build();

    var descriptor = converter.convert(symbol);

    Assertions.assertThat(descriptor.members()).hasSize(3);

    var membersByName = descriptor.members()
      .stream()
      .collect(Collectors.toMap(Descriptor::name, Function.identity()));

    Assertions.assertThat(membersByName).hasSize(3);
    Assertions.assertThat(membersByName.get("v1")).isInstanceOf(VariableDescriptor.class);
    Assertions.assertThat(membersByName.get("v2")).isNull();
    Assertions.assertThat(membersByName.get("foo1")).isInstanceOf(FunctionDescriptor.class);
    Assertions.assertThat(membersByName.get("foo2")).isNull();
    Assertions.assertThat(membersByName.get("overloaded_foo1")).isInstanceOf(AmbiguousDescriptor.class);
    Assertions.assertThat(membersByName.get("overloaded_foo2")).isNull();
  }

  @Test
  void builtinsTest() {
    var functionConverter = new FunctionSymbolToDescriptorConverter();
    var variableConverter = new VarSymbolToDescriptorConverter();
    var overloadedFunctionConverter = new OverloadedFunctionSymbolToDescriptorConverter(functionConverter);
    var converter = new ClassSymbolToDescriptorConverter(variableConverter, functionConverter, overloadedFunctionConverter, ProjectPythonVersion.currentVersionValues());

    var symbol = SymbolsProtos.ClassSymbol.newBuilder()
      .setName("int")
      .setFullyQualifiedName("builtins.int")
      .addSuperClasses("builtins.int_superclass")
      .setMetaclassName("builtins.meta_class")
      .build();

    var descriptor = converter.convert(symbol);

    Assertions.assertThat(descriptor.name()).isEqualTo("int");
    Assertions.assertThat(descriptor.fullyQualifiedName()).isEqualTo("int");
    Assertions.assertThat(descriptor.superClasses()).containsOnly("int_superclass");
    Assertions.assertThat(descriptor.metaclassFQN()).isEqualTo("meta_class");
  }

}
