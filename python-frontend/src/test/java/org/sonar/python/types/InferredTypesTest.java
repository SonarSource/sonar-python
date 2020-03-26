/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.types;

import org.junit.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.or;
import static org.sonar.python.types.InferredTypes.runtimeType;

public class InferredTypesTest {

  @Test
  public void test_runtimeType() {
    assertThat(runtimeType(null)).isEqualTo(anyType());
    assertThat(runtimeType(new SymbolImpl("b", "a.b"))).isEqualTo(anyType());
    ClassSymbol typeClass = new ClassSymbolImpl("b", "a.b");
    assertThat(runtimeType(typeClass)).isEqualTo(new RuntimeType(typeClass));
  }

  @Test
  public void test_or() {
    ClassSymbol a = new ClassSymbolImpl("a", "a");
    ClassSymbol b = new ClassSymbolImpl("b", "b");
    assertThat(or(anyType(), anyType())).isEqualTo(anyType());
    assertThat(or(anyType(), runtimeType(a))).isEqualTo(anyType());
    assertThat(or(runtimeType(a), anyType())).isEqualTo(anyType());
    assertThat(or(runtimeType(a), runtimeType(a))).isEqualTo(runtimeType(a));
    assertThat(or(runtimeType(a), runtimeType(b))).isNotEqualTo(anyType());
    assertThat(or(runtimeType(a), runtimeType(b))).isEqualTo(or(runtimeType(b), runtimeType(a)));
  }

  @Test
  public void test_aliased_type_annotations() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import List",
      "l : List[int]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation, TypeShed.builtinSymbols())).isEqualTo(InferredTypes.LIST);

    typeAnnotation = typeAnnotation(
      "from typing import Dict",
      "l : Dict[int, string]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation, TypeShed.builtinSymbols())).isEqualTo(InferredTypes.DICT);
  }

  @Test
  public void test_union_type_annotations() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[int, str]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation, TypeShed.builtinSymbols())).isEqualTo(InferredTypes.or(InferredTypes.INT, InferredTypes.STR));

    typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[int, str, bool]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation, TypeShed.builtinSymbols())).isEqualTo(InferredTypes.or(InferredTypes.or(InferredTypes.INT, InferredTypes.STR), InferredTypes.BOOL));

    typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[Union[int, str], bool]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation, TypeShed.builtinSymbols())).isEqualTo(InferredTypes.or(InferredTypes.or(InferredTypes.INT, InferredTypes.STR), InferredTypes.BOOL));

    typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[bool]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation, TypeShed.builtinSymbols())).isEqualTo(InferredTypes.BOOL);
  }

  @Test
  public void test_optional_type_annotations() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import Optional",
      "l : Optional[int]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation, TypeShed.builtinSymbols())).isEqualTo(InferredTypes.or(InferredTypes.INT, InferredTypes.NONE));

    typeAnnotation = typeAnnotation(
      "from typing import Optional",
      "l : Optional[int, string]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation, TypeShed.builtinSymbols())).isEqualTo(InferredTypes.anyType());
  }

  private TypeAnnotation typeAnnotation(String... code) {
    return PythonTestUtils.getLastDescendant(PythonTestUtils.parse(code), tree -> tree.is(Tree.Kind.VARIABLE_TYPE_ANNOTATION));
  }
}
