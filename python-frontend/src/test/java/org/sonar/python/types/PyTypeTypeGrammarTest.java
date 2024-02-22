/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.Objects;
import org.antlr.v4.runtime.RecognitionException;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Anything_typeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Builtin_typeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Class_typeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Generic_callable_typeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.TypeContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Type_listContext;
import org.sonar.python.types.pytype_grammar.PyTypeTypeGrammarParser.Union_typeContext;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class PyTypeTypeGrammarTest {

  @Test
  void test_parse_tree_string() {
    String typeString = "hello";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    PyTypeTypeGrammarParser.Qualified_typeContext qualifiedTypeContext = typeContext.qualified_type();
    assertThat(qualifiedTypeContext).isNotNull();

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromParseTree(typeContext);
    assertThat(typeFromParseTree.runtimeTypeSymbol().fullyQualifiedName()).isEqualTo("hello");
    assertThat(typeFromParseTree).isInstanceOf(RuntimeType.class);
  }

  @Test
  void test_class_1() {
    String typeString = "ClassType(GabaGool)";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Class_typeContext classType = typeContext.class_type();
    assertThat(classType).isNotNull();

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromParseTree(typeContext);
    assertThat(typeFromParseTree).isInstanceOf(RuntimeType.class);
    RuntimeType runtimeType = (RuntimeType) typeFromParseTree;
    assertThat(runtimeType.getTypeClass().name()).isEqualTo("GabaGool");
  }

  @Test
  void test_class_builtins_1() {
    String typeString = "ClassType(int)";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Class_typeContext classType = typeContext.class_type();
    assertThat(classType).isNotNull();

    Builtin_typeContext builtinTypeContext = classType.builtin_type();
    assertThat(builtinTypeContext).isNotNull();

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromParseTree(typeContext);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.INT);
  }

  @Test
  void test_class_builtins_2() {
    String typeString = "ClassType(builtins.int)";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Class_typeContext classType = typeContext.class_type();
    assertThat(classType).isNotNull();

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromParseTree(typeContext);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.INT);
  }

  @Test
  void test_class_builtins_3() {
    String typeString = "ClassType(builtins.dict)";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Class_typeContext classType = typeContext.class_type();
    assertThat(classType).isNotNull();

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromParseTree(typeContext);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.DICT);
  }

  @Test
  void test_none_type() {
    String typeString = "builtins.NoneType";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Builtin_typeContext builtinTypeContext = typeContext.builtin_type();
    assertThat(builtinTypeContext).isNotNull();

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromParseTree(typeContext);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.NONE);
  }

  @Test
  void test_anything_type() {
    String typeString = "AnythingType()";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Anything_typeContext anythingTypeContext = typeContext.anything_type();
    assertThat(anythingTypeContext).isNotNull();

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromParseTree(typeContext);
    assertThat(typeFromParseTree).isEqualTo(InferredTypes.anyType());
  }

  @Test
  void test_union_1() {

    String typeString = "UnionType(type_list=(hello, world))";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Union_typeContext unionType = typeContext.union_type();
    assertThat(unionType).isNotNull();

    Type_listContext typeListContext = unionType.type_list();

    assertThat(typeListContext.type(0).qualified_type().STRING(0)).hasToString("hello");
    assertThat(typeListContext.type(1).qualified_type().STRING(0)).hasToString("world");

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromParseTree(typeContext);
  }

  @Test
  void test_union_2() {
    String typeString = "UnionType(type_list=(gabagool, ova, here))";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Union_typeContext unionType = typeContext.union_type();
    assertThat(unionType).isNotNull();

    Type_listContext typeListContext = unionType.type_list();

    assertThat(typeListContext.type(0).qualified_type().STRING(0)).hasToString("gabagool");
    assertThat(typeListContext.type(1).qualified_type().STRING(0)).hasToString("ova");
    assertThat(typeListContext.type(2).qualified_type().STRING(0)).hasToString("here");
  }

  @Test
  void test_union_3() {

    String typeString = "UnionType(type_list=(ClassType(NoneElement), ClassType(IntegerElement)))";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Union_typeContext unionType = typeContext.union_type();
    assertThat(unionType).isNotNull();

    Type_listContext typeListContext = unionType.type_list();

    assertThat(typeListContext.type().stream().map(TypeContext::class_type).allMatch(Objects::nonNull)).isTrue();
  }

  @Test
  void test_callable_1() {
    String typeString = "GenericType(base_type=ClassType(typing.Callable), parameters=(AnythingType(), ClassType(builtins.NoneType)))";
    TypeContext typeContext = PyTypeTypeGrammar.getParseTree(typeString);

    Generic_callable_typeContext genericCallableTypeContext = typeContext.generic_callable_type();
    assertThat(genericCallableTypeContext).isNotNull();

    Type_listContext typeListContext = genericCallableTypeContext.type_list();
    assertThat(typeListContext).isNotNull();

    InferredType typeFromParseTree = PyTypeTypeGrammar.getTypeFromParseTree(typeContext);
    assertThat(typeFromParseTree.canHaveMember("__call__")).isTrue();

  }

  @Test
  void test_exception_1() {
    String someInvalidTypeString = "UnionType(type_list=(ClassType(None, ClassType(IntegerElement))";
    assertThatThrownBy(() -> PyTypeTypeGrammar.getParseTree(someInvalidTypeString)).isInstanceOf(RecognitionException.class);
  }

  @Test
  void test_exception_2() {
    String someInvalidTypeString = "str bool";
    assertThatThrownBy(() -> PyTypeTypeGrammar.getParseTree(someInvalidTypeString)).isInstanceOf(RecognitionException.class);
  }

  @Test
  void test_exception_3() {
    String someInvalidTypeString = "UnionType(type_list=(ClassType(None, ClassType(IntegerElement))";
    assertThatThrownBy(() -> PyTypeTypeGrammar.getParseTree(someInvalidTypeString)).isInstanceOf(RecognitionException.class);
  }
}
