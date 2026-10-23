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
package org.sonar.python.semantic.v2.typeshed;

import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.v2.GenericInfo.BareTypeArgumentStatus;
import org.sonar.plugins.python.api.types.v2.GenericInfo;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.index.AliasDescriptor;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.DescriptorUtils;
import org.sonar.python.index.DescriptorsToProtobuf;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.GenericInfoProtobuf;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.SymbolV2Impl;
import org.sonar.python.semantic.v2.converter.ClassDescriptorToPythonTypeConverter;
import org.sonar.python.semantic.v2.converter.ConversionContext;
import org.sonar.python.semantic.v2.converter.PythonTypeToDescriptorConverter;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.types.protobuf.SymbolsProtos;

class ClassSymbolToDescriptorConverterTest {

  @Test
  void genericInformationSurvivesDescriptorReconstruction() {
    var functionConverter = new FunctionSymbolToDescriptorConverter();
    var converter = new ClassSymbolToDescriptorConverter(new VarSymbolToDescriptorConverter(), functionConverter,
      new OverloadedFunctionSymbolToDescriptorConverter(functionConverter), Set.of("310"));
    var states = List.of(GenericInfo.UNKNOWN, GenericInfo.NON_GENERIC,
      new GenericInfo.Generic(BareTypeArgumentStatus.UNKNOWN),
      new GenericInfo.Generic(BareTypeArgumentStatus.CONTAINS_ANY),
      new GenericInfo.Generic(BareTypeArgumentStatus.NO_ANY));
    for (var state : states) {
      var symbol = SymbolsProtos.ClassSymbol.newBuilder().setName("Generic").setFullyQualifiedName("module.Generic")
        .setIsGeneric(state.isGeneric()).setGenericInfo(GenericInfoProtobuf.toProtobuf(state)).build();
      var descriptor = converter.convert(symbol);
      var restored = new ClassSymbolImpl(descriptor, "Reexported");
      Assertions.assertThat(restored.supportsGenerics()).isEqualTo(state.isGeneric());
      Assertions.assertThat(restored.genericInfo()).isEqualTo(state);

      var context = new ConversionContext("module", new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())),
        (ctx, value) -> PythonType.UNKNOWN, null);
      var type = new ClassDescriptorToPythonTypeConverter().convert(context, descriptor);
      var exported = new PythonTypeToDescriptorConverter().convert("reexport", new SymbolV2Impl("Exported"), Set.of(type));
      var protobuf = DescriptorsToProtobuf.toProtobufModuleDescriptor(Set.of(exported));
      var cached = (ClassDescriptor) DescriptorsToProtobuf.fromProtobuf(protobuf).iterator().next();
      Assertions.assertThat(cached.genericInfo()).isEqualTo(state);

      var alias = new AliasDescriptor("Alias", "other.Alias", cached);
      var aliasedSymbol = (ClassSymbolImpl) DescriptorUtils.symbolFromDescriptor(alias, ProjectLevelSymbolTable.empty(), null,
        new HashMap<>(), new HashMap<>());
      Assertions.assertThat(aliasedSymbol.genericInfo()).isEqualTo(state);
    }
  }

  @Test
  void test() {
    var functionConverter = new FunctionSymbolToDescriptorConverter();
    var variableConverter = new VarSymbolToDescriptorConverter();
    var overloadedFunctionConverter = new OverloadedFunctionSymbolToDescriptorConverter(functionConverter);
    var converter = new ClassSymbolToDescriptorConverter(
      variableConverter,
      functionConverter,
      overloadedFunctionConverter,
      Set.of("310", "311", "312", "313", "314"));

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

  @Test
  void builtinsTest() {
    var functionConverter = new FunctionSymbolToDescriptorConverter();
    var variableConverter = new VarSymbolToDescriptorConverter();
    var overloadedFunctionConverter = new OverloadedFunctionSymbolToDescriptorConverter(functionConverter);
    var converter = new ClassSymbolToDescriptorConverter(
      variableConverter,
      functionConverter,
      overloadedFunctionConverter,
      Set.of("310", "311", "312", "313", "314"));

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
