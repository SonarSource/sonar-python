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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.python.index.ClassDescriptor;
import org.sonar.python.index.FunctionDescriptor;
import org.sonar.python.index.TypeAnnotationDescriptor;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.semantic.v2.SymbolV2Impl;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verifyNoInteractions;

class FunctionDescriptorToPythonTypeConverterTest {
  @Test
  void unresolvedDecoratorsAreNotDroppedOrLookedUpAsNames() {
    var descriptor = new FunctionDescriptor.FunctionDescriptorBuilder().withName("overloaded")
      .withFullyQualifiedName("mod.overloaded").withHasDecorators(true)
      .withDecorators(Arrays.asList("typing.overload", "", null)).build();
    var table = spy(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    var context = new ConversionContext("mod", new LazyTypesContext(table), (c, d) -> PythonType.UNKNOWN, null);
    var function = (FunctionType) new FunctionDescriptorToPythonTypeConverter().convert(context, descriptor);
    assertThat(function.decorators()).hasSize(3);
    assertThat(function.decorators().get(1).type()).isSameAs(PythonType.UNKNOWN);
    assertThat(function.decorators().get(2).type()).isSameAs(PythonType.UNKNOWN);
    verifyNoInteractions(table);
    assertThat(function.decorators().getFirst().type()).isSameAs(table.getType("typing.overload"));
    var roundTrip = (FunctionDescriptor) new PythonTypeToDescriptorConverter().convert("mod", new SymbolV2Impl("overloaded"), Set.of(function));
    assertThat(roundTrip.decorators()).containsExactly("typing.overload", "", "");
    var restored = (FunctionType) new FunctionDescriptorToPythonTypeConverter().convert(context, roundTrip);
    assertThat(restored.decorators()).hasSize(3);
    assertThat(restored.decorators().get(1).type()).isSameAs(PythonType.UNKNOWN);
    assertThat(restored.decorators().get(2).type()).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void unsupportedClassTest() {
    var ctx = mock(ConversionContext.class);
    var descriptor = mock(ClassDescriptor.class);
    var converter = new FunctionDescriptorToPythonTypeConverter();
    Assertions.assertThatThrownBy(() -> converter.convert(ctx, descriptor))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessage("Unsupported Descriptor");
  }

  @Test
  void nullableCallableRetainsUncertaintyAndStableUnionMembership() {
    var callable = new TypeAnnotationDescriptor("Callable", TypeAnnotationDescriptor.TypeKind.CALLABLE, List.of(), null, false);
    var none = new TypeAnnotationDescriptor("None", TypeAnnotationDescriptor.TypeKind.NONE, List.of(), null, false);
    assertStableUnion(List.of(callable, none), TriBool.UNKNOWN);
  }

  @Test
  void genericUnionRetainsStableMembershipAfterResolvingAttributes() {
    var integer = new TypeAnnotationDescriptor("int", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "int", false);
    var string = new TypeAnnotationDescriptor("str", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "str", false);
    var integers = new TypeAnnotationDescriptor("list[int]", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(integer), "list", false);
    var strings = new TypeAnnotationDescriptor("set[str]", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(string), "set", false);
    assertStableUnion(List.of(integers, strings), TriBool.FALSE);
  }

  @Test
  void nestedGenericUnionRetainsStableMembership() {
    var integer = new TypeAnnotationDescriptor("int", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "int", false);
    var none = new TypeAnnotationDescriptor("None", TypeAnnotationDescriptor.TypeKind.NONE, List.of(), null, false);
    var optionalInteger = new TypeAnnotationDescriptor("int | None", TypeAnnotationDescriptor.TypeKind.UNION, List.of(integer, none), null, false);
    var values = new TypeAnnotationDescriptor("list[int | None]", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(optionalInteger), "list", false);
    assertStableUnion(List.of(values, none), TriBool.FALSE);
  }

  @Test
  void selfUnionRetainsStableMembership() {
    var self = new TypeAnnotationDescriptor("Self", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "list", true);
    var none = new TypeAnnotationDescriptor("None", TypeAnnotationDescriptor.TypeKind.NONE, List.of(), null, false);
    assertStableUnion(List.of(self, none), TriBool.UNKNOWN);
  }

  private static void assertStableUnion(List<TypeAnnotationDescriptor> alternatives, TriBool callability) {
    var annotation = new TypeAnnotationDescriptor("union", TypeAnnotationDescriptor.TypeKind.UNION, alternatives, null, false);
    var descriptor = new FunctionDescriptor.FunctionDescriptorBuilder().withName("value")
      .withFullyQualifiedName("mod.value").withTypeAnnotationDescriptor(annotation).build();
    var table = spy(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty()));
    var context = new ConversionContext("mod", new LazyTypesContext(table), (c, d) -> PythonType.UNKNOWN, null);
    var function = (FunctionType) new FunctionDescriptorToPythonTypeConverter().convert(context, descriptor);
    verifyNoInteractions(table);

    var union = (UnionType) function.returnType();
    var candidates = List.copyOf(union.candidates());
    int hashBeforeResolution = union.hashCode();
    var containingSet = new HashSet<>(Set.of(union));
    assertThat(union.hasMember("__call__")).isEqualTo(callability);
    candidates.forEach(FunctionDescriptorToPythonTypeConverterTest::inspectValueStructure);

    assertThat(union.hashCode()).isEqualTo(hashBeforeResolution);
    assertThat(union.candidates()).containsAll(candidates);
    assertThat(containingSet).contains(union);
    var reconstructed = UnionType.or(candidates);
    assertThat(union).isEqualTo(reconstructed);
    assertThat(reconstructed).isEqualTo(union);
    assertThat(function.returnType()).isEqualTo(union);
  }

  private static void inspectValueStructure(PythonType type) {
    if (type instanceof ObjectType objectType) {
      inspectValueStructure(objectType.unwrappedType());
      objectType.attributes().forEach(FunctionDescriptorToPythonTypeConverterTest::inspectValueStructure);
    } else if (type instanceof SelfType selfType) {
      inspectValueStructure(selfType.innerType());
    } else if (type instanceof UnionType unionType) {
      unionType.candidates().forEach(FunctionDescriptorToPythonTypeConverterTest::inspectValueStructure);
    }
  }

  @Test
  void unionReturnTypesPreserveValueShapeWithoutEagerResolution() {
    var intType = new TypeAnnotationDescriptor("int", TypeAnnotationDescriptor.TypeKind.INSTANCE, List.of(), "int", false);
    var noneType = new TypeAnnotationDescriptor("None", TypeAnnotationDescriptor.TypeKind.NONE, List.of(), null, false);
    var annotation = new TypeAnnotationDescriptor("int | None", TypeAnnotationDescriptor.TypeKind.UNION,
      List.of(intType, noneType), null, false);
    var descriptor = new FunctionDescriptor.FunctionDescriptorBuilder().withName("value")
      .withFullyQualifiedName("mod.value").withTypeAnnotationDescriptor(annotation).build();
    var converter = new FunctionDescriptorToPythonTypeConverter();
    var table = mock(ProjectLevelTypeTable.class);
    var lazyContext = new ConversionContext("mod", new LazyTypesContext(table), (c, d) -> PythonType.UNKNOWN, null);
    converter.convert(lazyContext, descriptor);
    verifyNoInteractions(table);

    var context = new ConversionContext("mod", new LazyTypesContext(new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty())),
      (c, d) -> PythonType.UNKNOWN, null);
    var result = ((FunctionType) converter.convert(context, descriptor)).returnType();
    assertThat(result).isInstanceOfSatisfying(UnionType.class,
      union -> assertThat(union.candidates()).hasSize(2).allSatisfy(candidate -> assertThat(candidate).isInstanceOf(ObjectType.class)));
    assertThat(result.hasMember("__call__")).isEqualTo(TriBool.FALSE);
  }

}
