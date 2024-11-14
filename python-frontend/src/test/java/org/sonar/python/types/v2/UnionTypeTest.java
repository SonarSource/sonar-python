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
package org.sonar.python.types.v2;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.assertj.core.api.InstanceOfAssertFactories;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.semantic.v2.LazyTypesContext;
import org.sonar.python.types.v2.UnknownType.UnresolvedImportType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.python.types.v2.TypesTestUtils.BOOL_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.STR_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;


class UnionTypeTest {

  @Test
  void basicUnion() {
    FileInput fileInput = parseAndInferTypes("42;\"hello\"");
    PythonType intType = ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0).typeV2();
    PythonType strType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();

    UnionType unionType = new UnionType(Set.of(intType, strType));

    assertThat(unionType.isCompatibleWith(intType)).isTrue();
    assertThat(unionType.isCompatibleWith(strType)).isTrue();

    assertThat(unionType.displayName()).contains("Union[int, str]");
    assertThat(unionType.unwrappedType()).isEqualTo(unionType);
    assertThat(unionType.instanceDisplayName()).isEmpty();
    assertThat(unionType.definitionLocation()).isEmpty();
  }

  @Test
  void unionWithUnknown() {
    FileInput fileInput = parseAndInferTypes("42;foo()");
    PythonType intType = ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0).typeV2();
    PythonType strType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    UnionType unionType = new UnionType(Set.of(intType, strType));

    assertThat(unionType.displayName()).isEmpty();
    assertThat(unionType.instanceDisplayName()).isEmpty();
  }

  @Test
  void or_with_itself() {
    PythonType unionType = UnionType.or(INT_TYPE, INT_TYPE);
    assertThat(unionType).isEqualTo(INT_TYPE);
  }

  @Test
  void or_with_union_type() {
    PythonType unionType = UnionType.or(INT_TYPE, STR_TYPE);
    PythonType result = UnionType.or(unionType, BOOL_TYPE);
    assertThat(((UnionType) result).candidates()).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE, BOOL_TYPE);
  }

  @Test
  void or_with_null() {
    PythonType type = UnionType.or(INT_TYPE, null);
    assertThat(type).isEqualTo(INT_TYPE);
    type = UnionType.or(null, INT_TYPE);
    assertThat(type).isEqualTo(INT_TYPE);
  }

  @Test
  void or_unionType() {
    FileInput fileInput = parseAndInferTypes("42;\"hello\"");
    PythonType intType = ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0).typeV2();
    PythonType strType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();

    UnionType unionType = (UnionType) UnionType.or(intType, strType);

    assertThat(unionType.candidates()).containsExactlyInAnyOrder(intType, strType);
    assertThat(unionType.displayName()).contains("Union[int, str]");

    PythonType unknownUnion = UnionType.or(intType, PythonType.UNKNOWN);
    assertThat(unknownUnion).isEqualTo(PythonType.UNKNOWN);

    unknownUnion = UnionType.or(PythonType.UNKNOWN, intType);
    assertThat(unknownUnion).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void or_unresolevdImportType()  {
    var unresolvedImportType = new UnresolvedImportType("unknown");
    var unresolvedImportType2 = new UnresolvedImportType("unknown2");
    var unionType = UnionType.or(unresolvedImportType, unresolvedImportType2);

    assertThat(unionType).isInstanceOf(UnionType.class);
    assertThat(((UnionType) unionType).candidates()).containsExactlyInAnyOrder(unresolvedImportType, unresolvedImportType2);
  }

  @Test
  void hasMemberUnionType() {
    FileInput fileInput = parseAndInferTypes("""
      class MyClass:
        def common(): ...
        def foo(): ...
      
      class MyOtherClass:
        def common(): ...
        def bar(): ...
      
      if cond():
        x = MyClass()
      else:
        x = MyOtherClass()
      x
      """);
    var lastExpressionStatement = (ExpressionStatement) fileInput.statements().statements().get(fileInput.statements().statements().size() -1);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().hasMember("common")).isEqualTo(TriBool.TRUE);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().hasMember("foo")).isEqualTo(TriBool.UNKNOWN);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().hasMember("bar")).isEqualTo(TriBool.UNKNOWN);
    assertThat(lastExpressionStatement.expressions().get(0).typeV2().hasMember("qix")).isEqualTo(TriBool.FALSE);
  }

  @Test
  void unionTypeCandidatesResolution() {
    var candidate1 = new ClassType("candidate", "mymod.candidate", Set.of(
      new Member("a", new FunctionTypeBuilder("a").build())
    ), List.of(), List.of(), List.of(), false, null);

    var candidate2 = new ClassType("candidate", "mymod.candidate", Set.of(
      new Member("a", new FunctionTypeBuilder("a").build())
    ), List.of(), List.of(), List.of(), false, null);

    var unionType = UnionType.or(candidate1, candidate2);
    // TODO: SONARPY-1901 UnionType.or should resolve it to class type since the candidates are the same
    assertThat(unionType).isInstanceOf(UnionType.class)
      .extracting(UnionType.class::cast)
      .extracting(UnionType::candidates)
      .extracting(ArrayList::new)
      .asInstanceOf(InstanceOfAssertFactories.LIST)
      .hasSize(2)
      .contains(candidate1, candidate2);
  }

  @Test
  void noLazyTypeInUnionType() {
    PythonType lazyType = new LazyType("foo", Mockito.mock(LazyTypesContext.class));
    assertThatThrownBy(() -> UnionType.or(INT_TYPE, lazyType))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessage("UnionType cannot contain Lazy types");
  }
}
