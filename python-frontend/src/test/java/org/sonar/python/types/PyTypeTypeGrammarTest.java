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
package org.sonar.python.types;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;

class PyTypeTypeGrammarTest {

  @Test
  void test_parse_tree_string_number_and_underscore() {
    String typeString = "hello_3";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree.runtimeTypeSymbol().fullyQualifiedName()).isEqualTo("hello_3");
    assertThat(typeFromParseTree).isInstanceOf(RuntimeType.class);
  }

  @Test
  void test_parse_tree_string() {
    String typeString = "hello";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree.runtimeTypeSymbol().fullyQualifiedName()).isEqualTo("hello");
    assertThat(typeFromParseTree).isInstanceOf(RuntimeType.class);
  }

  @Test
  void test_class_1() {
    String typeString = "ClassType(GabaGool)";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isInstanceOf(RuntimeType.class);
    RuntimeType runtimeType = (RuntimeType) typeFromParseTree;
    assertThat(runtimeType.getTypeClass().name()).isEqualTo("GabaGool");
  }

  @Test
  void test_class_builtins_2() {
    String typeString = "ClassType(builtins.int)";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.INT);
  }

  @Test
  void test_class_builtins_3() {
    String typeString = "ClassType(builtins.dict)";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.DICT);
  }

  @Test
  void test_class_builtins_4() {
    String typeString = "builtins.range";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isEqualTo(new RuntimeType("range"));
  }

  @Test
  void test_class_builtins_generator() {
    String typeString = "builtins.generator";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isEqualTo(new RuntimeType(new ClassSymbolImpl("Generator", "typing.Generator")));
  }

  @Test
  void test_none_type() {
    String typeString = "builtins.NoneType";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.NONE);
  }

  @Test
  void test_anything_type() {
    String typeString = "AnythingType()";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.anyType());
  }

  @Test
  void test_nothing_type() {
    String typeString = "NothingType()";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.anyType());
  }

  @Test
  void test_union_3() {
    String typeString = "UnionType(type_list=(ClassType(builtins.int), ClassType(builtins.str)))";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isEqualTo(UnionType.or(InferredTypes.INT, InferredTypes.STR));
  }

  @Test
  void test_callable_1() {
    String typeString = "GenericType(base_type=ClassType(typing.Callable), parameters=(AnythingType(), ClassType(builtins.NoneType)))";

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isInstanceOf(RuntimeType.class);
    RuntimeType runtimeType = (RuntimeType) typeFromParseTree;
    assertThat(runtimeType.getTypeClass().name()).isEqualTo("Callable");

  }

  @Test
  void test_generics_builtins() {
    String typeString = "GenericType(base_type=ClassType(builtins.type), parameters=(ClassType(builtins.ImportError),))";
    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromString(typeString);
    assertThat(typeFromParseTree).isInstanceOf(RuntimeType.class);
    RuntimeType runtimeType = (RuntimeType) typeFromParseTree;
    assertThat(runtimeType.getTypeClass().name()).isEqualTo("ImportError");

  }
}
