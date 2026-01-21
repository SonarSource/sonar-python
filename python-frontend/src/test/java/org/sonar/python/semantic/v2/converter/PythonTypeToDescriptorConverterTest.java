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
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.Member;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.SymbolV2Impl;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.LazyUnionType;
import org.sonar.python.types.v2.SpecialFormType;
import org.sonar.python.types.v2.TypesTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class PythonTypeToDescriptorConverterTest {

  private final PythonTypeToDescriptorConverter converter = new PythonTypeToDescriptorConverter();
  private final TypeWrapper intTypeWrapper = TypeWrapper.of(new UnknownType.UnresolvedImportType("int"));
  private final TypeWrapper floatTypeWrapper = TypeWrapper.of(new UnknownType.UnresolvedImportType("float"));
  private final LocationInFile location = new LocationInFile("myFile", 1, 2, 3, 4);

  @Test
  void testConvertFunctionTypeWithoutDecorator() {
    ParameterV2 parameterV2 = new ParameterV2("param", TypeWrapper.of(intTypeWrapper.type()), false, true, false, false, false, location);
    FunctionType functionType = new FunctionType("functionType",
      "my_package.foo.functionType",
      List.of(new ModuleType("bar")),
      List.of(parameterV2),
      List.of(TypeWrapper.of(new UnknownType.UnresolvedImportType("abc.abstractmethod"))),
      floatTypeWrapper,
      TypeOrigin.LOCAL,
      true,
      true,
      true,
      false,
      false,
      null,
      location);
    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("myFunction"), Set.of(functionType));

    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor functionDescriptor = (FunctionDescriptor) descriptor;
    assertThat(functionDescriptor.name()).isEqualTo("functionType");
    assertThat(functionDescriptor.fullyQualifiedName()).isEqualTo("my_package.foo.functionType");
    assertThat(functionDescriptor.kind()).isEqualTo(Descriptor.Kind.FUNCTION);
    assertThat(functionDescriptor.isAsynchronous()).isTrue();
    assertThat(functionDescriptor.isInstanceMethod()).isTrue();

    assertThat(functionDescriptor.annotatedReturnTypeName()).isEqualTo("float");
    assertThat(functionDescriptor.typeAnnotationDescriptor()).isNotNull();
    assertThat(functionDescriptor.typeAnnotationDescriptor().fullyQualifiedName()).isEqualTo("float");
    assertThat(functionDescriptor.typeAnnotationDescriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(functionDescriptor.typeAnnotationDescriptor().isSelf()).isFalse();

    assertThat(functionDescriptor.hasDecorators()).isTrue();
    assertThat(functionDescriptor.decorators()).isNotEmpty().containsOnly("abc.abstractmethod");
    assertThat(functionDescriptor.definitionLocation()).isEqualTo(location);

    assertThat(functionDescriptor.parameters()).hasSize(1);
    FunctionDescriptor.Parameter parameter = functionDescriptor.parameters().get(0);
    assertThat(parameter.name()).isEqualTo("param");
    assertThat(parameter.annotatedType()).isEqualTo("int");
    assertThat(parameter.hasDefaultValue()).isFalse();
    assertThat(parameter.isKeywordOnly()).isTrue();
    assertThat(parameter.isPositionalOnly()).isFalse();
    assertThat(parameter.isKeywordVariadic()).isFalse();
    assertThat(parameter.isPositionalVariadic()).isFalse();
    assertThat(parameter.location()).isEqualTo(location);
    assertThat(parameter.descriptor()).isNotNull();
    assertThat(parameter.descriptor().fullyQualifiedName()).isEqualTo("int");
  }

  @Test
  void testConvertClassType() {
    ClassType classType = new ClassType("classType", "my_package.classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper),
      List.of(intTypeWrapper.type()), true, false, location);
    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("myClass"), Set.of(classType));

    assertThat(descriptor).isInstanceOf(ClassDescriptor.class);
    ClassDescriptor classDescriptor = (ClassDescriptor) descriptor;
    assertThat(classDescriptor.name()).isEqualTo("myClass");
    assertThat(classDescriptor.fullyQualifiedName()).isEqualTo("foo.myClass");
    assertThat(classDescriptor.kind()).isEqualTo(Descriptor.Kind.CLASS);
    assertThat(classDescriptor.superClasses()).containsExactly("float");

    assertThat(classDescriptor.members()).hasSize(1);
    Descriptor memberDescriptor = classDescriptor.members().iterator().next();
    assertThat(memberDescriptor).isInstanceOf(VariableDescriptor.class);
    VariableDescriptor memberVariableDescriptor = (VariableDescriptor) memberDescriptor;
    assertThat(memberVariableDescriptor.name()).isEqualTo("aMember");
    assertThat(memberVariableDescriptor.annotatedType()).isEqualTo("int");
    assertThat(memberVariableDescriptor.fullyQualifiedName()).isEqualTo("foo.myClass.aMember");

    assertThat(classDescriptor.hasDecorators()).isTrue();
    assertThat(classDescriptor.definitionLocation()).isEqualTo(location);
    assertThat(classDescriptor.hasMetaClass()).isTrue();

    // SONARPY-2307 support for superClass is missing in ClassType
    assertThat(classDescriptor.hasSuperClassWithoutDescriptor()).isFalse();
    assertThat(classDescriptor.metaclassFQN()).isEqualTo("int");
    // SONARPY-2307 support for generics is missing in ClassType
    assertThat(classDescriptor.supportsGenerics()).isFalse();
  }

  @Test
  void testConvertUnresolvedImportType() {
    UnknownType.UnresolvedImportType unresolvedImportType = new UnknownType.UnresolvedImportType("anImport");
    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("myImportedType"), Set.of(unresolvedImportType));

    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    VariableDescriptor variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myImportedType");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myImportedType");
    assertThat(variableDescriptor.annotatedType()).isEqualTo("anImport");
  }

  @Test
  void testConvertOtherType() {
    LazyType lazyType = new LazyType("foo", new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())));
    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("myLazySymbol"), Set.of(lazyType));
    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    VariableDescriptor variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myLazySymbol");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myLazySymbol");
    assertThat(variableDescriptor.annotatedType()).isNull();

    ModuleType moduleType = new ModuleType("myModule");
    descriptor = converter.convert("foo", new SymbolV2Impl("myModulSymbol"), Set.of(moduleType));
    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myModulSymbol");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myModulSymbol");
    assertThat(variableDescriptor.annotatedType()).isNull();

    ObjectType objectType = ObjectType.fromType(lazyType);
    descriptor = converter.convert("foo", new SymbolV2Impl("myObjectSymbol"), Set.of(objectType));
    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myObjectSymbol");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myObjectSymbol");
    assertThat(variableDescriptor.annotatedType()).isNull();

    LazyUnionType lazyUnionType = (LazyUnionType) LazyUnionType.or(Set.of(lazyType, objectType));
    descriptor = converter.convert("foo", new SymbolV2Impl("myLazyUnionSymbol"), Set.of(lazyUnionType));
    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myLazyUnionSymbol");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myLazyUnionSymbol");
    assertThat(variableDescriptor.annotatedType()).isNull();
  }

  @Test
  void testConvertUnionType() {
    ClassType classType = new ClassType("classType", "my_package.classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper),
      List.of(intTypeWrapper.type()), true, false, location);
    ClassType anotherClassType = new ClassType("classType", "my_package.classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper),
      List.of(intTypeWrapper.type()), true, false, location);
    PythonType unionType = UnionType.or(classType, anotherClassType);
    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("myUnionType"), Set.of(unionType));

    assertThat(descriptor).isInstanceOf(AmbiguousDescriptor.class);
    AmbiguousDescriptor ambiguousDescriptor = (AmbiguousDescriptor) descriptor;
    assertThat(ambiguousDescriptor.name()).isEqualTo("myUnionType");
    assertThat(ambiguousDescriptor.fullyQualifiedName()).isEqualTo("foo.myUnionType");
    assertThat(ambiguousDescriptor.kind()).isEqualTo(Descriptor.Kind.AMBIGUOUS);
    // SONARPY-2307 the two class types in the union are rigorously the same but the converter creates an ambigouous symbol
    assertThat(ambiguousDescriptor.alternatives()).hasSize(2);
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::name).containsExactlyInAnyOrder("myUnionType", "myUnionType");
    assertThat(ambiguousDescriptor.alternatives()).extracting(Object::getClass).allMatch(c -> c == ClassDescriptor.class);
  }

  @Test
  void testConvertManyTypes() {
    ClassType classType = new ClassType("classType", "my_package.classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper),
      List.of(intTypeWrapper.type()), true, false, location);
    FunctionType functionType = new FunctionType("functionType", "my_package.functionType", List.of(new ModuleType("bar")), List.of(), List.of(), floatTypeWrapper,
      TypeOrigin.LOCAL, true, false, true, false, false, null, location);
    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("myUnionType"), Set.of(functionType, classType));

    assertThat(descriptor).isInstanceOf(AmbiguousDescriptor.class);
    AmbiguousDescriptor ambiguousDescriptor = (AmbiguousDescriptor) descriptor;
    assertThat(ambiguousDescriptor.name()).isEqualTo("myUnionType");
    assertThat(ambiguousDescriptor.fullyQualifiedName()).isEqualTo("foo.myUnionType");
    assertThat(ambiguousDescriptor.kind()).isEqualTo(Descriptor.Kind.AMBIGUOUS);
    assertThat(ambiguousDescriptor.alternatives()).hasSize(2);
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::kind).containsExactlyInAnyOrder(Descriptor.Kind.CLASS, Descriptor.Kind.FUNCTION);
  }

  @Test
  void testConvertManyTypesWithUnionType() {
    ClassType classType = new ClassType("classType", "my_package.classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper),
      List.of(intTypeWrapper.type()), true, false, location);
    ClassType anotherClassType = new ClassType("classType", "my_package.classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper),
      List.of(intTypeWrapper.type()), true, false, location);

    PythonType unionType = UnionType.or(classType, anotherClassType);
    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("myUnionType"), Set.of(unionType, classType));

    assertThat(descriptor).isInstanceOf(AmbiguousDescriptor.class);
    AmbiguousDescriptor ambiguousDescriptor = (AmbiguousDescriptor) descriptor;
    assertThat(ambiguousDescriptor.name()).isEqualTo("myUnionType");
    assertThat(ambiguousDescriptor.fullyQualifiedName()).isEqualTo("foo.myUnionType");
    assertThat(ambiguousDescriptor.kind()).isEqualTo(Descriptor.Kind.AMBIGUOUS);
    // SONARPY-2307 the two class types in the union are rigorously the same but the converter creates an ambigouous descriptor
    assertThat(ambiguousDescriptor.alternatives()).hasSize(3);
    assertThat(ambiguousDescriptor.alternatives()).extracting(Descriptor::name).allMatch(s -> s.equals("myUnionType"));
    assertThat(ambiguousDescriptor.alternatives()).extracting(Object::getClass).allMatch(c -> c == ClassDescriptor.class);
  }

  @Test
  void testReassignedFunctionIsNotConverted() {
    FileInput fileInput = TypesTestUtils.parseAndInferTypes("""
      def foo(): ...
      my_var = foo
      foo
      my_var
      """);
    Name fooName = (Name) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0);
    PythonType fooType = fooName.typeV2();
    SymbolV2 fooSymbol = fooName.symbolV2();
    Name myVarName = (Name) ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0);
    PythonType myVarType = myVarName.typeV2();
    SymbolV2 myVarSymbol = myVarName.symbolV2();

    Descriptor fooDescriptor = converter.convert("mod", fooSymbol, Set.of(fooType));
    Descriptor myVarDescriptor = converter.convert("mod", myVarSymbol, Set.of(myVarType));
    assertThat(fooDescriptor).isInstanceOf(FunctionDescriptor.class);
    assertThat(myVarDescriptor).isInstanceOf(VariableDescriptor.class);
    assertThat(((VariableDescriptor) myVarDescriptor)).satisfies(d -> {
      assertThat(d.name()).isEqualTo("my_var");
      assertThat(d.fullyQualifiedName()).isEqualTo("mod.my_var");
      assertThat(d.annotatedType()).isNull();
    });
  }

  @Test
  void testMissingCandidatesThrowsException() {
    SymbolV2 symbol = new SymbolV2Impl("mySymbol");
    Set<PythonType> emptySet = Set.of();
    assertThatThrownBy(() -> converter.convert("foo", symbol, emptySet))
      .isInstanceOf(IllegalStateException.class)
      .hasMessage("No candidate found for descriptor mySymbol");
  }

  @Test
  void testConvertSelfType() {
    ClassType classType = new ClassType("MyClass", "my_package.MyClass", Set.of(), List.of(), List.of(), List.of(), false, false, location);
    PythonType selfType = SelfType.of(classType);

    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("mySelfSymbol"), Set.of(selfType));

    assertThat(descriptor).isInstanceOf(ClassDescriptor.class);
    ClassDescriptor classDescriptor = (ClassDescriptor) descriptor;
    assertThat(classDescriptor.isSelf()).isTrue();
    assertThat(classDescriptor.name()).isEqualTo("mySelfSymbol");
    assertThat(classDescriptor.fullyQualifiedName()).isEqualTo("foo.mySelfSymbol");
  }

  @Test
  void testConvertFunctionParametersWithTypeAnnotationDescriptors() {
    ClassType stringClassType = new ClassType("str", "builtins.str", Set.of(), List.of(), List.of(), List.of(), false, false, location);
    ClassType myClassType = new ClassType("MyClass", "my_package.MyClass", Set.of(), List.of(), List.of(), List.of(), false, false, location);
    PythonType selfType = SelfType.of(myClassType);
    ObjectType objectType = ObjectType.fromType(stringClassType);
    FunctionType callableType = new FunctionType("my_function", "my_package.my_function", List.of(), List.of(), List.of(), floatTypeWrapper, TypeOrigin.LOCAL, false, false, false,
      false, false, null, location);

    List<ParameterV2> parameters = List.of(
      new ParameterV2("self_param", TypeWrapper.of(selfType), false, false, false, false, false, location),
      new ParameterV2("class_param", TypeWrapper.of(stringClassType), false, false, false, false, false, location),
      new ParameterV2("object_param", TypeWrapper.of(objectType), false, false, false, false, false, location),
      new ParameterV2("callable_param", TypeWrapper.of(callableType), false, false, false, false, false, location));

    FunctionType functionType = new FunctionType("test_function",
      "my_package.test_function",
      List.of(),
      parameters,
      List.of(),
      floatTypeWrapper,
      TypeOrigin.LOCAL,
      false,
      false,
      false,
      false,
      false,
      null,
      location);

    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("test_function"), Set.of(functionType));

    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor functionDescriptor = (FunctionDescriptor) descriptor;
    assertThat(functionDescriptor.parameters()).hasSize(4);

    FunctionDescriptor.Parameter selfParam = functionDescriptor.parameters().get(0);
    assertThat(selfParam.name()).isEqualTo("self_param");
    assertThat(selfParam.descriptor()).isNotNull();
    assertThat(selfParam.descriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(selfParam.descriptor().isSelf()).isTrue();
    assertThat(selfParam.descriptor().fullyQualifiedName()).isEqualTo("my_package.MyClass");

    FunctionDescriptor.Parameter classParam = functionDescriptor.parameters().get(1);
    assertThat(classParam.name()).isEqualTo("class_param");
    assertThat(classParam.descriptor()).isNotNull();
    assertThat(classParam.descriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(classParam.descriptor().isSelf()).isFalse();
    assertThat(classParam.descriptor().fullyQualifiedName()).isEqualTo("builtins.str");

    FunctionDescriptor.Parameter objectParam = functionDescriptor.parameters().get(2);
    assertThat(objectParam.name()).isEqualTo("object_param");
    assertThat(objectParam.descriptor()).isNotNull();
    assertThat(objectParam.descriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(objectParam.descriptor().isSelf()).isFalse();
    assertThat(objectParam.descriptor().fullyQualifiedName()).isEqualTo("builtins.str");

    FunctionDescriptor.Parameter callableParam = functionDescriptor.parameters().get(3);
    assertThat(callableParam.name()).isEqualTo("callable_param");
    assertThat(callableParam.descriptor()).isNotNull();
    assertThat(callableParam.descriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.CALLABLE);
    assertThat(callableParam.descriptor().isSelf()).isFalse();
    assertThat(callableParam.descriptor().fullyQualifiedName()).isEqualTo("my_package.my_function");
  }

  @Test
  void testConvertParameterTypesWithEdgeCases() {

    ModuleType moduleType = new ModuleType("my_module", "my_package.my_module", null, java.util.Map.of());
    UnknownType.UnresolvedImportType unresolvedImportType = new UnknownType.UnresolvedImportType("unresolved.import.path");
    
    SpecialFormType specialFormType = new SpecialFormType("special_form_type");
    List<ParameterV2> parameters = List.of(
      new ParameterV2("module_param", TypeWrapper.of(moduleType), false, false, false, false, false, location),
      new ParameterV2("unresolved_param", TypeWrapper.of(unresolvedImportType), false, false, false, false, false, location),
      new ParameterV2("special_form", TypeWrapper.of(specialFormType), false, false, false, false, false, location)
    );

    FunctionType testFunction = new FunctionType("test_function",
      "my_package.test_function",
      List.of(),
      parameters,
      List.of(),
      floatTypeWrapper,
      TypeOrigin.LOCAL,
      false,
      false,
      false,
      false,
      false,
      null,
      location);

    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("test_function"), Set.of(testFunction));

    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor functionDescriptor = (FunctionDescriptor) descriptor;
    assertThat(functionDescriptor.parameters()).hasSize(3);


    // ModuleType case - should return FQN but TypeAnnotationDescriptor is null (not supported in createTypeAnnotationDescriptor)
    FunctionDescriptor.Parameter moduleParam = functionDescriptor.parameters().get(0);
    assertThat(moduleParam.name()).isEqualTo("module_param");
    assertThat(moduleParam.annotatedType()).isEqualTo("my_package.my_module");
    assertThat(moduleParam.descriptor()).isNull();

    FunctionDescriptor.Parameter unresolvedParam = functionDescriptor.parameters().get(1);
    assertThat(unresolvedParam.name()).isEqualTo("unresolved_param");
    assertThat(unresolvedParam.annotatedType()).isEqualTo("unresolved.import.path");
    assertThat(unresolvedParam.descriptor()).isNotNull();
    assertThat(unresolvedParam.descriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(unresolvedParam.descriptor().fullyQualifiedName()).isEqualTo("unresolved.import.path");

    FunctionDescriptor.Parameter specialFormParam = functionDescriptor.parameters().get(2);
    assertThat(specialFormParam.name()).isEqualTo("special_form");
    assertThat(specialFormParam.annotatedType()).isEqualTo("special_form_type");
    assertThat(specialFormParam.descriptor()).isNull();
  }

  @Test
  void testConvertFunctionReturnTypeWithClassType() {
    ClassType stringClassType = new ClassType("str", "builtins.str", Set.of(), List.of(), List.of(), List.of(), false, false, location);
    
    FunctionType funcWithClassReturn = new FunctionType("func1",
      "my_package.func1",
      List.of(),
      List.of(),
      List.of(),
      TypeWrapper.of(stringClassType),
      TypeOrigin.LOCAL,
      false,
      false,
      false,
      false,
      false,
        null,
      location);

    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("func1"), Set.of(funcWithClassReturn));
    
    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor funcDesc = (FunctionDescriptor) descriptor;
    assertThat(funcDesc.annotatedReturnTypeName()).isEqualTo("builtins.str");
    assertThat(funcDesc.typeAnnotationDescriptor()).isNotNull();
    assertThat(funcDesc.typeAnnotationDescriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(funcDesc.typeAnnotationDescriptor().fullyQualifiedName()).isEqualTo("builtins.str");
    assertThat(funcDesc.typeAnnotationDescriptor().isSelf()).isFalse();
  }

  @Test
  void testConvertFunctionReturnTypeWithSelfType() {
    ClassType myClassType = new ClassType("MyClass", "my_package.MyClass", Set.of(), List.of(), List.of(), List.of(), false, false, location);
    PythonType selfType = SelfType.of(myClassType);

    FunctionType funcWithSelfReturn = new FunctionType("func2",
      "my_package.func2",
      List.of(),
      List.of(),
      List.of(),
      TypeWrapper.of(selfType),
      TypeOrigin.LOCAL,
      false,
      false,
      false,
      false,
      false,
        null,
      location);

    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("func2"), Set.of(funcWithSelfReturn));
    
    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor funcDesc = (FunctionDescriptor) descriptor;
    assertThat(funcDesc.annotatedReturnTypeName()).isEqualTo("my_package.MyClass");
    assertThat(funcDesc.typeAnnotationDescriptor()).isNotNull();
    assertThat(funcDesc.typeAnnotationDescriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(funcDesc.typeAnnotationDescriptor().fullyQualifiedName()).isEqualTo("my_package.MyClass");
    assertThat(funcDesc.typeAnnotationDescriptor().isSelf()).isTrue();
  }

  @Test
  void testConvertFunctionReturnTypeWithCallableType() {
    FunctionType callableType = new FunctionType("my_function", "my_package.my_function", List.of(), List.of(), List.of(), floatTypeWrapper, TypeOrigin.LOCAL, false, false, false,
      false, false, null, location);

    FunctionType funcWithCallableReturn = new FunctionType("func3",
      "my_package.func3",
      List.of(),
      List.of(),
      List.of(),
      TypeWrapper.of(callableType),
      TypeOrigin.LOCAL,
      false,
      false,
      false,
      false,
      false,
      null,
      location);

    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("func3"), Set.of(funcWithCallableReturn));
    
    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor funcDesc = (FunctionDescriptor) descriptor;
    assertThat(funcDesc.annotatedReturnTypeName()).isEqualTo("my_package.my_function");
    assertThat(funcDesc.typeAnnotationDescriptor()).isNotNull();
    assertThat(funcDesc.typeAnnotationDescriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.CALLABLE);
    assertThat(funcDesc.typeAnnotationDescriptor().fullyQualifiedName()).isEqualTo("my_package.my_function");
    assertThat(funcDesc.typeAnnotationDescriptor().isSelf()).isFalse();
  }

  @Test
  void testConvertFunctionReturnTypeWithUnresolvedImportType() {
    UnknownType.UnresolvedImportType unresolvedImportType = new UnknownType.UnresolvedImportType("unresolved.import.path");

    FunctionType funcWithUnresolvedReturn = new FunctionType("func4",
      "my_package.func4",
      List.of(),
      List.of(),
      List.of(),
      TypeWrapper.of(unresolvedImportType),
      TypeOrigin.LOCAL,
      false,
      false,
      false,
      false,
      false,
      null,
      location);

    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("func4"), Set.of(funcWithUnresolvedReturn));
    
    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor funcDesc = (FunctionDescriptor) descriptor;
    assertThat(funcDesc.annotatedReturnTypeName()).isEqualTo("unresolved.import.path");
    assertThat(funcDesc.typeAnnotationDescriptor()).isNotNull();
    assertThat(funcDesc.typeAnnotationDescriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(funcDesc.typeAnnotationDescriptor().fullyQualifiedName()).isEqualTo("unresolved.import.path");
    assertThat(funcDesc.typeAnnotationDescriptor().isSelf()).isFalse();
  }

  @Test
  void testConvertFunctionReturnTypeWithObjectTypeOfClassType() {
    ClassType stringClassType = new ClassType("str", "builtins.str", Set.of(), List.of(), List.of(), List.of(), false, false, location);
    ObjectType objectType = ObjectType.fromType(stringClassType);
    
    FunctionType funcWithObjectReturn = new FunctionType("func5",
      "my_package.func5",
      List.of(),
      List.of(),
      List.of(),
      TypeWrapper.of(objectType),
      TypeOrigin.LOCAL,
      false,
      false,
      false,
      false,
      false,
      null,
      location);

    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("func5"), Set.of(funcWithObjectReturn));
    
    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor funcDesc = (FunctionDescriptor) descriptor;
    assertThat(funcDesc.annotatedReturnTypeName()).isEqualTo("builtins.str");
    assertThat(funcDesc.typeAnnotationDescriptor()).isNotNull();
    assertThat(funcDesc.typeAnnotationDescriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(funcDesc.typeAnnotationDescriptor().fullyQualifiedName()).isEqualTo("builtins.str");
    assertThat(funcDesc.typeAnnotationDescriptor().isSelf()).isFalse();
  }

  @Test
  void testConvertFunctionReturnTypeWithObjectTypeOfSelfType() {
    ClassType myClassType = new ClassType("MyClass", "my_package.MyClass", Set.of(), List.of(), List.of(), List.of(), false, false, location);
    PythonType selfType = SelfType.of(myClassType);
    ObjectType objectTypeOfSelfType = ObjectType.fromType(selfType);
    
    FunctionType funcWithObjectSelfReturn = new FunctionType("func6",
      "my_package.func6",
      List.of(),
      List.of(),
      List.of(),
      TypeWrapper.of(objectTypeOfSelfType),
      TypeOrigin.LOCAL,
      false,
      false,
      false,
      false,
      false,
      null,
      location);

    Descriptor descriptor = converter.convert("foo", new SymbolV2Impl("func6"), Set.of(funcWithObjectSelfReturn));
    
    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor funcDesc = (FunctionDescriptor) descriptor;
    assertThat(funcDesc.annotatedReturnTypeName()).isEqualTo("my_package.MyClass");
    assertThat(funcDesc.typeAnnotationDescriptor()).isNotNull();
    assertThat(funcDesc.typeAnnotationDescriptor().kind()).isEqualTo(TypeAnnotationDescriptor.TypeKind.INSTANCE);
    assertThat(funcDesc.typeAnnotationDescriptor().fullyQualifiedName()).isEqualTo("my_package.MyClass");
    assertThat(funcDesc.typeAnnotationDescriptor().isSelf()).isTrue();
  }
}
