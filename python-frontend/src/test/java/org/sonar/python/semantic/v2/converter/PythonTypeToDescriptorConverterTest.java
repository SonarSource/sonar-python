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
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.index.AmbiguousDescriptor;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.LazyUnionType;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.ParameterV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeOrigin;
import org.sonar.python.types.v2.TypeWrapper;
import org.sonar.python.types.v2.TypesTestUtils;
import org.sonar.python.types.v2.UnionType;
import org.sonar.python.types.v2.UnknownType;

import static org.assertj.core.api.Assertions.assertThat;

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
      null,
      location);
    Descriptor descriptor = converter.convert("foo", new SymbolV2("myFunction"), Set.of(functionType));

    assertThat(descriptor).isInstanceOf(FunctionDescriptor.class);
    FunctionDescriptor functionDescriptor = (FunctionDescriptor) descriptor;
    assertThat(functionDescriptor.name()).isEqualTo("functionType");
    assertThat(functionDescriptor.fullyQualifiedName()).isEqualTo("my_package.foo.functionType");
    assertThat(functionDescriptor.kind()).isEqualTo(Descriptor.Kind.FUNCTION);
    assertThat(functionDescriptor.isAsynchronous()).isTrue();
    assertThat(functionDescriptor.isInstanceMethod()).isTrue();

    // SONARPY-2306 support for return type is missing in FunctionType
    assertThat(functionDescriptor.annotatedReturnTypeName()).isNull();

    // SONARPY-2306 support for type annotation is missing in FunctionType
    assertThat(functionDescriptor.typeAnnotationDescriptor()).isNull();

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
  }

  @Test
  void testConvertClassType() {
    ClassType classType = new ClassType("classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper), List.of(intTypeWrapper.type()), true, location);
    Descriptor descriptor = converter.convert("foo", new SymbolV2("myClass"), Set.of(classType));

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
    // SONARPY-2307 support for metaclassFQN is missing in ClassType
    assertThat(classDescriptor.metaclassFQN()).isNull();
    // SONARPY-2307 support for generics is missing in ClassType
    assertThat(classDescriptor.supportsGenerics()).isFalse();
  }

  @Test
  void testConvertUnresolvedImportType() {
    UnknownType.UnresolvedImportType unresolvedImportType = new UnknownType.UnresolvedImportType("anImport");
    Descriptor descriptor = converter.convert("foo", new SymbolV2("myImportedType"), Set.of(unresolvedImportType));

    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    VariableDescriptor variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myImportedType");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myImportedType");
    assertThat(variableDescriptor.annotatedType()).isEqualTo("anImport");
  }

  @Test
  void testConvertOtherType() {
    LazyType lazyType = new LazyType("foo", new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())));
    Descriptor descriptor = converter.convert("foo", new SymbolV2("myLazySymbol"), Set.of(lazyType));
    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    VariableDescriptor variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myLazySymbol");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myLazySymbol");
    assertThat(variableDescriptor.annotatedType()).isNull();

    ModuleType moduleType = new ModuleType("myModule");
    descriptor = converter.convert("foo", new SymbolV2("myModulSymbol"), Set.of(moduleType));
    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myModulSymbol");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myModulSymbol");
    assertThat(variableDescriptor.annotatedType()).isNull();

    ObjectType objectType = new ObjectType(lazyType);
    descriptor = converter.convert("foo", new SymbolV2("myObjectSymbol"), Set.of(objectType));
    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myObjectSymbol");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myObjectSymbol");
    assertThat(variableDescriptor.annotatedType()).isNull();

    LazyUnionType lazyUnionType = new LazyUnionType(Set.of(lazyType, objectType));
    descriptor = converter.convert("foo", new SymbolV2("myLazyUnionSymbol"), Set.of(lazyUnionType));
    assertThat(descriptor).isInstanceOf(VariableDescriptor.class);
    variableDescriptor = (VariableDescriptor) descriptor;
    assertThat(variableDescriptor.name()).isEqualTo("myLazyUnionSymbol");
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("foo.myLazyUnionSymbol");
    assertThat(variableDescriptor.annotatedType()).isNull();
  }

  @Test
  void testConvertUnionType() {
    ClassType classType = new ClassType("classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper), List.of(intTypeWrapper.type()), true, location);
    ClassType anotherClassType = new ClassType("classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper), List.of(intTypeWrapper.type()), true, location);
    UnionType unionType = new UnionType(Set.of(classType, anotherClassType));
    Descriptor descriptor = converter.convert("foo", new SymbolV2("myUnionType"), Set.of(unionType));

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
    ClassType classType = new ClassType("classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper), List.of(intTypeWrapper.type()), true, location);
    FunctionType functionType = new FunctionType("functionType", "my_package.functionType", List.of(new ModuleType("bar")), List.of(), List.of(), floatTypeWrapper, TypeOrigin.LOCAL, true, false, true, false, null, location);
    Descriptor descriptor = converter.convert("foo", new SymbolV2("myUnionType"), Set.of(functionType, classType));

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
    ClassType classType = new ClassType("classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper), List.of(intTypeWrapper.type()), true, location);
    ClassType anotherClassType = new ClassType("classType", Set.of(new Member("aMember", intTypeWrapper.type())), List.of(), List.of(floatTypeWrapper), List.of(intTypeWrapper.type()), true, location);

    UnionType unionType = new UnionType(Set.of(classType, anotherClassType));
    Descriptor descriptor = converter.convert("foo", new SymbolV2("myUnionType"), Set.of(unionType, classType));

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

}
