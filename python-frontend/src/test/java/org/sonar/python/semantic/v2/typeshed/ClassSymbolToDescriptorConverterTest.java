package org.sonar.python.semantic.v2.typeshed;

import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
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
    var converter = new ClassSymbolToDescriptorConverter(variableConverter, functionConverter, overloadedFunctionConverter);

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
  }
  
}
