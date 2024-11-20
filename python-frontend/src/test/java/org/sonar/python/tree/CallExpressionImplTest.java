/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.tree;

import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.types.DeclaredType;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.getLastDescendant;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.lastExpressionInFunction;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.types.InferredTypes.DECL_INT;
import static org.sonar.python.types.InferredTypes.DECL_STR;
import static org.sonar.python.types.InferredTypes.DICT;
import static org.sonar.python.types.InferredTypes.LIST;
import static org.sonar.python.types.InferredTypes.SET;
import static org.sonar.python.types.InferredTypes.TUPLE;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.typeName;

class CallExpressionImplTest {

  @Test
  void constructor_type() {
    FileInput fileInput = parse(
      "class A: ...",
      "A()");

    ClassDef classDef = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
    CallExpression call = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CALL_EXPR));

    assertThat(call.type()).isEqualTo(InferredTypes.runtimeType(classDef.name().symbol()));
  }

  @Test
  void function_call_type() {
    assertThat(lastExpression("len()").type()).isEqualTo(InferredTypes.INT);
  }

  @Test
  void ambiguous_callee_symbol_type() {
    FileInput fileInput = parse(
      "class A: ...",
      "class A: ...",
      "A()");

    ClassDef classDef = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
    CallExpression call = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CALL_EXPR));

    AmbiguousSymbol ambiguousSymbol = (AmbiguousSymbol) classDef.name().symbol();
    Iterator<Symbol> iterator = ambiguousSymbol.alternatives().iterator();
    InferredType firstClassType = InferredTypes.runtimeType(iterator.next());
    InferredType secondClassType = InferredTypes.runtimeType(iterator.next());
    assertThat(call.type()).isEqualTo(InferredTypes.or(firstClassType, secondClassType));
    assertThat(call.type()).isEqualTo(firstClassType);
    assertThat(call.type()).isEqualTo(secondClassType);

    fileInput = parse(
      "class Base: ...",
      "class A(Base): ...",
      "class A: ...",
      "A()");

    classDef = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
    call = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CALL_EXPR));

    ambiguousSymbol = (AmbiguousSymbol) classDef.name().symbol();
    iterator = ambiguousSymbol.alternatives().iterator();
    firstClassType = InferredTypes.runtimeType(iterator.next());
    secondClassType = InferredTypes.runtimeType(iterator.next());
    assertThat(call.type()).isEqualTo(InferredTypes.or(firstClassType, secondClassType));
    assertThat(call.type()).isNotEqualTo(firstClassType);
    assertThat(call.type()).isNotEqualTo(secondClassType);
  }

  @Test
  void not_callable_callee_symbol_type() {
    assertThat(lastExpression(
      "x = 42",
      "x()"
    ).type()).isEqualTo(InferredTypes.anyType());
  }

  @Test
  void typing_NamedTuple_call_type() {
    assertThat(lastExpression(
      "from typing import NamedTuple",
      "NamedTuple()"
    ).type()).isEqualTo(InferredTypes.TYPE);
  }

  @Test
  void null_callee_symbol_type() {
    assertThat(lastExpression("x()").type()).isEqualTo(InferredTypes.anyType());
  }

  @Test
  void method_call_with_declared_type() {
    assertThat(lastExpression(
      "def foo(param: str):",
      "  x = param.capitalize()",
      "  x"
    ).type()).isEqualTo(DECL_STR);

    assertThat(lastExpression(
      "def foo(param: A):",
      "  x = param.capitalize()",
      "  x"
    ).type()).isEqualTo(anyType());

    assertThat(lastExpression(
      "def foo(param):",
      "  x = param.capitalize()",
      "  x"
    ).type()).isEqualTo(anyType());

    // TODO: handle member access dependencies for declared type (SONARPY-781)
    assertThat(lastExpression(
      "def foo(param: str):",
      "  x = param.capitalize().capitalize()",
      "  x"
    ).type()).isEqualTo(anyType());

    assertThat(lastExpression(
      "class A:",
      "  def meth() -> int: ...",
      "def foo(param: A):",
      "  x = param.meth()",
      "  x"
    ).type()).isEqualTo(DECL_INT);
  }

  @Test
  void method_call_with_declared_type_union_and_optional() {
    ClassSymbolImpl union = new ClassSymbolImpl("Union", "typing.Union");
    List<DeclaredType> typeArgs = Stream.of(DECL_INT, DECL_STR).map(DeclaredType.class::cast).toList();
    assertThat(lastExpression(
      "from typing import Union",
      "class A:",
      "  def meth() -> Union[int, str]: ...",
      "def foo(param: A):",
      "  x = param.meth()",
      "  x"
    ).type()).isEqualTo(new DeclaredType(union, typeArgs));

    // TODO: handle UnionType of annotation coming from typeshed (SONARPY-782)
    assertThat(lastExpression(
      // object.__reduce__ returns 'Union[str, Tuple[Any, ...]]'
      "def foo(param: object):",
      "  x = param.__reduce__()",
      "  x"
    ).type()).isEqualTo(anyType());

    ClassSymbolImpl optional = new ClassSymbolImpl("Optional", "typing.Optional");
    typeArgs = Stream.of(DECL_STR).map(DeclaredType.class::cast).toList();
    assertThat(lastExpression(
      "from typing import Optional",
      "class A:",
      "  def meth() -> Optional[str]: ...",
      "def foo(param: A):",
      "  x = param.meth()",
      "  x"
    ).type()).isEqualTo(new DeclaredType(optional, typeArgs));

    assertThat(lastExpression(
      "from typing import Optional",
      "class A:",
      "  def meth() -> str: ...",
      "def foo(param: Optional[A]):",
      "  x = param.meth()",
      "  x"
    ).type()).isEqualTo(DECL_STR);
  }

  @Test
  void inner_class_constructor_declared_type() {
    InferredType type = lastExpression(
      "class A:",
      "  class B: ...",
      "def foo(param: A):",
      "  x = param.B()",
      "  x"
    ).type();
    assertThat(type).isInstanceOf(DeclaredType.class);
    assertThat(typeName(type)).isEqualTo("B");
  }

  @Test
  void test_generic_collections() {
    assertThat(lastExpressionInFunction("list[int]()").type()).isEqualTo(LIST);
    assertThat(lastExpressionInFunction("tuple[int]()").type()).isEqualTo(TUPLE);
    assertThat(lastExpressionInFunction("dict[str, int]()").type()).isEqualTo(DICT);
    assertThat(lastExpressionInFunction("set[int]()").type()).isEqualTo(SET);
    assertThat(lastExpressionInFunction("frozenset[int]()").type().canOnlyBe("frozenset")).isTrue();
    assertThat(lastExpressionInFunction("from collections import deque; deque[int]()").type().canOnlyBe("collections.deque")).isTrue();
    assertThat(lastExpressionInFunction("from collections import defaultdict; defaultdict[str, int]()").type().canOnlyBe("collections.defaultdict")).isTrue();
    assertThat(lastExpressionInFunction("from collections import OrderedDict; OrderedDict[str, int]()").type().canOnlyBe("collections.OrderedDict")).isTrue();
    assertThat(lastExpressionInFunction("from collections import Counter; Counter[int]()").type().canOnlyBe("collections.Counter")).isTrue();
    assertThat(lastExpressionInFunction("from collections import ChainMap; ChainMap[str, int]()").type().canOnlyBe("collections.ChainMap")).isTrue();
    assertThat(lastExpressionInFunction("from foo import bar; bar[str]()").type()).isEqualTo(anyType());
    assertThat(lastExpressionInFunction("int[str]()").type()).isEqualTo(anyType());
  }

}
