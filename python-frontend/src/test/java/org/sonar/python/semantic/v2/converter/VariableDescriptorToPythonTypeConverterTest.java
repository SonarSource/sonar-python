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
package org.sonar.python.semantic.v2.converter;

import java.util.List;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.types.v2.TypeUtils;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.types.v2.LazyType;
import org.sonar.python.types.v2.SpecialFormType;
import org.sonar.python.types.v2.LazyUnionType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.sonar.python.types.v2.TypesTestUtils.lastName;

class VariableDescriptorToPythonTypeConverterTest {
  @Test
  void unsupportedClassTest() {
    var ctx = mock(ConversionContext.class);
    var descriptor = mock(ClassDescriptor.class);
    var converter = new VariableDescriptorToPythonTypeConverter();
    Assertions.assertThatThrownBy(() -> converter.convert(ctx, descriptor))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessage("Unsupported Descriptor");
  }

  @Test
  void testSpecialFormTypeConversion() {
    var ctx = mock(ConversionContext.class);
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
  void testTypeAliasResolvesToLazyType() {
    var lazyTypesContext = new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    var converter = new VariableDescriptorToPythonTypeConverter();

    // TypeAlias vars (e.g. typing.Text = str, socket.error = OSError) must produce a LazyType
    // for the target class, not an ObjectType (an instance). Without this, calling
    // socket.error(...) would be flagged by S5756 as calling a non-callable instance.
    VariableDescriptor typeAliasDesc = new VariableDescriptor(
      "Text", "typing.Text", "builtins.str", false, true, List.of(), List.of()
    );
    var ctx = new ConversionContext("typing", lazyTypesContext, (c, d) -> PythonType.UNKNOWN, null);

    PythonType result = converter.convert(ctx, typeAliasDesc);

    assertThat(result)
      .as("TypeAlias var must produce a LazyType for the target class, not ObjectType (instance)")
      .isInstanceOfSatisfying(LazyType.class,
        lazyType -> assertThat(lazyType.importPath()).isEqualTo("builtins.str"));
  }

  @Test
  void testNonTypeAliasVariableProducesObjectType() {
    var lazyTypesContext = new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    var converter = new VariableDescriptorToPythonTypeConverter();

    // A regular variable with a type annotation must still produce ObjectType (instance semantics).
    // Regression guard: the is_type_alias branch must not affect non-alias vars.
    VariableDescriptor regularDesc = new VariableDescriptor(
      "x", "mod.x", "builtins.str", false, false, List.of(), List.of()
    );
    var ctx = new ConversionContext("mod", lazyTypesContext, (c, d) -> PythonType.UNKNOWN, null);

    PythonType result = converter.convert(ctx, regularDesc);

    assertThat(result)
      .as("Regular variable with a type annotation must remain an ObjectType (instance)")
      .isInstanceOf(ObjectType.class);
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
  @Test
  void unionAnnotationsRemainLazyDuringConversion() {
    var typeTable = mock(ProjectLevelTypeTable.class);
    var context = new ConversionContext("mod", new LazyTypesContext(typeTable), (c, d) -> PythonType.UNKNOWN, null);
    var annotation = unionAnnotation(instanceAnnotation("int"), instanceAnnotation("str"));
    var descriptor = new VariableDescriptor("value", "mod.value", null, false, false, List.of(), List.of(), annotation);

    assertThat(new VariableDescriptorToPythonTypeConverter().convert(context, descriptor)).isInstanceOf(LazyUnionType.class);
    verifyNoInteractions(typeTable);
  }

  @Test
  void unionAnnotationsPreserveInstanceAndCallableCandidates() {
    var typeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    var context = new ConversionContext("mod", new LazyTypesContext(typeTable), (c, d) -> PythonType.UNKNOWN, null);
    var none = new TypeAnnotationDescriptor("None", TypeAnnotationDescriptor.TypeKind.NONE, List.of(), null, false);
    var callable = new TypeAnnotationDescriptor("Callable", TypeAnnotationDescriptor.TypeKind.CALLABLE, List.of(), null, false);
    var converter = new VariableDescriptorToPythonTypeConverter();

    var annotations = List.of(unionAnnotation(instanceAnnotation("int"), instanceAnnotation("str")),
      unionAnnotation(instanceAnnotation("int"), none), unionAnnotation(instanceAnnotation("int"), callable));
    var expectedNames = List.of("str", "NoneType", "function");
    for (int i = 0; i < annotations.size(); i++) {
      var annotation = annotations.get(i);
      var expectedName = expectedNames.get(i);
      var descriptor = new VariableDescriptor("value", "mod.value", null, false, false, List.of(), List.of(), annotation);
      var result = TypeUtils.resolved(converter.convert(context, descriptor));
      assertThat(result).isInstanceOfSatisfying(UnionType.class, union -> {
        assertThat(union.candidates()).extracting(candidate -> ((ClassType) candidate.unwrappedType()).name())
          .containsExactlyInAnyOrder("int", expectedName);
        assertThat(union.candidates()).anySatisfy(candidate -> {
          assertThat(candidate).isInstanceOf(ObjectType.class);
          assertThat(((ClassType) candidate.unwrappedType()).name()).isEqualTo("int");
        });
        assertThat(union.candidates()).allSatisfy(candidate -> assertThat(candidate).isInstanceOf(ObjectType.class));
        assertThat(union.hasMember("__call__")).isEqualTo("function".equals(expectedName) ? TriBool.UNKNOWN : TriBool.FALSE);
      });
    }
  }

  @Test
  void osErrorErrnoRemainsNonCallableAcrossStubVersions() {
    var errno = lastName("e = OSError()", "e.errno()").typeV2();
    // The previous stdlib encoded errno as int; current typeshed uses int | None.
    assertThat(errno).isNotSameAs(PythonType.UNKNOWN);
    assertThat(errno.hasMember("__call__")).isEqualTo(TriBool.FALSE);
    var type = TypeUtils.resolved(errno);
    if (type instanceof UnionType union) {
      assertThat(union.candidates()).allSatisfy(candidate -> assertThat(candidate).isInstanceOf(ObjectType.class));
      assertThat(union.candidates()).extracting(candidate -> ((ClassType) candidate.unwrappedType()).name())
        .containsExactlyInAnyOrder("int", "NoneType");
    } else {
      assertThat(type).isInstanceOf(ObjectType.class);
      assertThat(((ClassType) type.unwrappedType()).name()).isEqualTo("int");
    }
  }

  private static TypeAnnotationDescriptor instanceAnnotation(String fqn) {
    return new TypeAnnotationDescriptor(fqn, TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), fqn, false);
  }

  private static TypeAnnotationDescriptor unionAnnotation(TypeAnnotationDescriptor... candidates) {
    return new TypeAnnotationDescriptor("union", TypeAnnotationDescriptor.TypeKind.UNION, List.of(candidates), null, false);
  }

}
