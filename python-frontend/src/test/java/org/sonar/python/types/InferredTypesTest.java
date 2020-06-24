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

import java.util.Arrays;
import java.util.HashSet;
import org.junit.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.InferredTypes.INT;
import static org.sonar.python.types.InferredTypes.STR;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.or;
import static org.sonar.python.types.InferredTypes.runtimeType;
import static org.sonar.python.types.TypeShed.typeShedClass;

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
  public void ambiguous_class_symbol() {
    ClassSymbol a1 = new ClassSymbolImpl("a", "mod1.a");
    ClassSymbol a2 = new ClassSymbolImpl("a", "mod2.a");
    Symbol ambiguous = AmbiguousSymbolImpl.create(new HashSet<>(Arrays.asList(a1, a2)));
    assertThat(runtimeType(ambiguous)).isEqualTo(or(runtimeType(a1), runtimeType(a2)));
  }

  @Test
  public void test_aliased_type_annotations() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import List",
      "l : List[int]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation)).isEqualTo(InferredTypes.LIST);

    typeAnnotation = typeAnnotation(
      "from typing import Dict",
      "l : Dict[int, string]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation)).isEqualTo(InferredTypes.DICT);
  }

  @Test
  public void test_union_type_annotations() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[int, str]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation)).isEqualTo(InferredTypes.or(InferredTypes.INT, InferredTypes.STR));

    typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[int, str, bool]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation)).isEqualTo(InferredTypes.or(InferredTypes.or(InferredTypes.INT, InferredTypes.STR), InferredTypes.BOOL));

    typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[Union[int, str], bool]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation)).isEqualTo(InferredTypes.or(InferredTypes.or(InferredTypes.INT, InferredTypes.STR), InferredTypes.BOOL));

    typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[bool]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation)).isEqualTo(InferredTypes.BOOL);
  }

  @Test
  public void test_optional_type_annotations() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import Optional",
      "l : Optional[int]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation)).isEqualTo(InferredTypes.or(InferredTypes.INT, InferredTypes.NONE));

    typeAnnotation = typeAnnotation(
      "from typing import Optional",
      "l : Optional[int, string]"
    );
    assertThat(InferredTypes.declaredType(typeAnnotation)).isEqualTo(InferredTypes.anyType());
  }

  @Test
  public void test_typeSymbol() {
    assertThat(InferredTypes.typeSymbols(STR)).containsExactly(typeShedClass("str"));

    ClassSymbol a = new ClassSymbolImpl("A", "mod.A");
    assertThat(InferredTypes.typeSymbols(new RuntimeType(a))).containsExactly(a);

    assertThat(InferredTypes.typeSymbols(or(STR, INT))).containsExactlyInAnyOrder(typeShedClass("str"), typeShedClass("int"));
    assertThat(InferredTypes.typeSymbols(InferredTypes.anyType())).isEmpty();
  }

  @Test
  public void test_typeName() {
    assertThat(InferredTypes.typeName(STR)).isEqualTo("str");

    ClassSymbol a = new ClassSymbolImpl("A", "mod.A");
    assertThat(InferredTypes.typeName(new RuntimeType(a))).isEqualTo("A");

    assertThat(InferredTypes.typeName(or(STR, INT))).isNull();
    assertThat(InferredTypes.typeName(InferredTypes.anyType())).isNull();
  }

  @Test
  public void test_typeLocation() {
    assertThat(InferredTypes.typeClassLocation(STR)).isNull();

    LocationInFile locationA = new LocationInFile("foo.py", 1, 1, 1, 1);
    RuntimeType aType = new RuntimeType(new ClassSymbolImpl("A", "mod.A", locationA, false, false, null));
    assertThat(InferredTypes.typeClassLocation(aType)).isEqualTo(locationA);

    LocationInFile locationB = new LocationInFile("foo.py", 1, 2, 1, 2);
    RuntimeType bType = new RuntimeType(new ClassSymbolImpl("B", "mod.B", locationB, false, false, null));
    assertThat(InferredTypes.typeClassLocation(or(aType, bType))).isNull();
    assertThat(InferredTypes.typeClassLocation(InferredTypes.anyType())).isNull();
  }

  private TypeAnnotation typeAnnotation(String... code) {
    return PythonTestUtils.getLastDescendant(PythonTestUtils.parse(code), tree -> tree.is(Tree.Kind.VARIABLE_TYPE_ANNOTATION));
  }
}
