/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.python.types;

import com.google.protobuf.TextFormat;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.AmbiguousSymbolImpl;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.lastExpressionInFunction;
import static org.sonar.python.types.InferredTypes.COMPLEX;
import static org.sonar.python.types.InferredTypes.DECL_INT;
import static org.sonar.python.types.InferredTypes.DECL_STR;
import static org.sonar.python.types.InferredTypes.DECL_TYPE;
import static org.sonar.python.types.InferredTypes.DICT;
import static org.sonar.python.types.InferredTypes.FLOAT;
import static org.sonar.python.types.InferredTypes.INT;
import static org.sonar.python.types.InferredTypes.LIST;
import static org.sonar.python.types.InferredTypes.NONE;
import static org.sonar.python.types.InferredTypes.SET;
import static org.sonar.python.types.InferredTypes.STR;
import static org.sonar.python.types.InferredTypes.TUPLE;
import static org.sonar.python.types.InferredTypes.TYPE;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.containsDeclaredType;
import static org.sonar.python.types.InferredTypes.fromTypeAnnotation;
import static org.sonar.python.types.InferredTypes.fromTypeshedProtobuf;
import static org.sonar.python.types.InferredTypes.fromTypeshedTypeAnnotation;
import static org.sonar.python.types.InferredTypes.getBuiltinCategory;
import static org.sonar.python.types.InferredTypes.getBuiltinsTypeCategory;
import static org.sonar.python.types.InferredTypes.isDeclaredTypeWithTypeClass;
import static org.sonar.python.types.InferredTypes.or;
import static org.sonar.python.types.InferredTypes.runtimeType;
import static org.sonar.python.types.InferredTypes.typeName;
import static org.sonar.python.types.InferredTypes.typeSymbols;
import static org.sonar.python.types.TypeShed.typeShedClass;

class InferredTypesTest {

  @Test
  void test_runtimeType() {
    assertThat(runtimeType(null)).isEqualTo(anyType());
    assertThat(runtimeType(new SymbolImpl("b", "a.b"))).isEqualTo(anyType());
    ClassSymbol typeClass = new ClassSymbolImpl("b", "a.b");
    assertThat(runtimeType(typeClass)).isEqualTo(new RuntimeType(typeClass));
  }

  @Test
  void test_declaredType() {
    assertThat(InferredTypes.typeName(declaredType(new SymbolImpl("b", "a.b")))).isEqualTo("b");
    ClassSymbol typeClass = new ClassSymbolImpl("b", "a.b");
    assertThat(declaredType(typeClass).canOnlyBe("a.b")).isFalse();
    assertThat(declaredType(typeClass).canBeOrExtend("a.b")).isTrue();

    ClassSymbol typeClass1 = new ClassSymbolImpl("b", "a1.b");
    ClassSymbol typeClass2 = new ClassSymbolImpl("b", "a2.b");
    InferredType declaredType = declaredType(AmbiguousSymbolImpl.create(typeClass1, typeClass2));
    assertThat(declaredType.canBeOrExtend("a1.b")).isTrue();
    assertThat(declaredType.canBeOrExtend("a2.b")).isTrue();
    assertThat(declaredType.canOnlyBe("a1.b")).isFalse();

    assertThat(fromTypeAnnotation(typeAnnotation("x: unknown"))).isEqualTo(anyType());
  }

  @Test
  void test_or() {
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
  void ambiguous_class_symbol() {
    ClassSymbol a1 = new ClassSymbolImpl("a", "mod1.a");
    ClassSymbol a2 = new ClassSymbolImpl("a", "mod2.a");
    Symbol ambiguous = AmbiguousSymbolImpl.create(new HashSet<>(Arrays.asList(a1, a2)));
    assertThat(runtimeType(ambiguous)).isEqualTo(or(runtimeType(a1), runtimeType(a2)));
  }

  @Test
  void test_aliased_type_annotations() {
    assertAliasedTypeAnnotation(BuiltinTypes.LIST,
      "from typing import List",
      "l : List[int]");

    assertAliasedTypeAnnotation(BuiltinTypes.TUPLE,
      "from typing import Tuple",
      "l : Tuple[int]"
    );

    assertAliasedTypeAnnotation(BuiltinTypes.DICT,
      "from typing import Dict",
      "l : Dict[int, str]"
    );

    assertAliasedTypeAnnotation(BuiltinTypes.SET,
      "from typing import Set",
      "l : Set[int]"
    );

    assertAliasedTypeAnnotation("frozenset",
      "from typing import FrozenSet",
      "l : FrozenSet[int]"
    );

    assertAliasedTypeAnnotation("type",
      "from typing import Type",
      "l : Type[int]"
    );

    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import DefaultDict",
      "l : DefaultDict[int, str]"
    );
    InferredType type = fromTypeAnnotation(typeAnnotation);
    assertThat(typeName(type)).isEqualTo("DefaultDict[int, str]");

    typeAnnotation = typeAnnotation(
      "from typing import Deque",
      "l : Deque[int]"
    );
    type = fromTypeAnnotation(typeAnnotation);
    assertThat(typeName(type)).isEqualTo("Deque[int]");

    typeAnnotation = typeAnnotation(
      "from typing import List",
      "l : List"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(LIST);
    assertThat(((DeclaredType) fromTypeAnnotation(typeAnnotation)).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName).containsExactly("list");
  }

  @Test
  void test_tuples_classes_are_compatibles() {
    TypeAnnotation tupleAnnotation = typeAnnotation(
      "t1 : tuple"
    );
    InferredType tupleType = fromTypeshedTypeAnnotation(tupleAnnotation);
    ClassSymbolImpl typeClass = new ClassSymbolImpl("MyTuple", "mod.MyTuple");
    typeClass.addSuperClass(tupleType.runtimeTypeSymbol());
    typeClass.addMembers(List.of(new ClassSymbolImpl("SomeMember", "mod.MyTuple.SomeMember")));
    assertThat(tupleType.isCompatibleWith(declaredType(typeClass))).isTrue();
  }

  private void assertAliasedTypeAnnotation(String type, String... code) {
    TypeAnnotation typeAnnotation = typeAnnotation(code);
    ClassSymbol typeClass = typeShedClass(type);
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(runtimeType(typeClass));
    assertThat(typeSymbols(fromTypeAnnotation(typeAnnotation))).containsExactly(typeClass);
  }

  @Test
  void test_union_type_annotations() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[int, str]"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(InferredTypes.or(InferredTypes.INT, InferredTypes.STR));
    InferredType declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(DeclaredType.class);
    assertThat(((DeclaredType) declaredType).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder("int", "str");

    typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[int, str, bool]"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(InferredTypes.or(InferredTypes.or(InferredTypes.INT, InferredTypes.STR), InferredTypes.BOOL));
    declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(DeclaredType.class);
    assertThat(((DeclaredType) declaredType).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder("int", "str", "bool");

    typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[Union[int, str], bool]"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(InferredTypes.or(InferredTypes.or(InferredTypes.INT, InferredTypes.STR), InferredTypes.BOOL));
    declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(DeclaredType.class);
    assertThat(((DeclaredType) declaredType).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder("int", "str", "bool");

    typeAnnotation = typeAnnotation(
      "from typing import Union",
      "l : Union[bool]"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(InferredTypes.BOOL);
    declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(DeclaredType.class);
    assertThat(((DeclaredType) declaredType).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder("bool");
  }

  @Test
  void test_annotated_type_annotation() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import Annotated",
      "x : Annotated[str, y]"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(STR);
    assertThat(fromTypeAnnotation(typeAnnotation)).isEqualTo(DECL_STR);
  }

  @Test
  void test_text_annotation() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import Text",
      "l : Text"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(STR);
    InferredType declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(DeclaredType.class);
    assertThat(((DeclaredType) declaredType).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder("str");
  }

  @Test
  void test_none_annotation() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "l : None"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(NONE);
    InferredType declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(DeclaredType.class);
    assertThat(((DeclaredType) declaredType).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder("NoneType");
  }
 
  @Test
  void test_any_annotation() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import Any",
      "l : Any"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(anyType());
    InferredType declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(AnyType.class);
  }

  @Test
  void test_unknown_annotation() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "l : MyType"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(anyType());
    InferredType declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(AnyType.class);
  }

  @Test
  void test_type_annotation() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "l : type"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(TYPE);
    assertThat(fromTypeAnnotation(typeAnnotation)).isEqualTo(DECL_TYPE);
  }
 

  @Test
  void test_optional_type_annotations() {
    TypeAnnotation typeAnnotation = typeAnnotation(
      "from typing import Optional",
      "l : Optional[int]"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(InferredTypes.or(InferredTypes.INT, NONE));
    InferredType declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(DeclaredType.class);
    assertThat(((DeclaredType) declaredType).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder("NoneType", "int");

    typeAnnotation = typeAnnotation(
      "from typing import Optional",
      "l : Optional[int, str]"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(InferredTypes.anyType());
    declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(DeclaredType.class);
    assertThat(((DeclaredType) declaredType).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName)
      .containsExactlyInAnyOrder("typing.Optional");

    typeAnnotation = typeAnnotation(
      "from typing import Optional",
      "l : Optional[int, unknown_symbol]"
    );
    assertThat(fromTypeshedTypeAnnotation(typeAnnotation)).isEqualTo(InferredTypes.anyType());
    declaredType = fromTypeAnnotation(typeAnnotation);
    assertThat(declaredType).isInstanceOf(AnyType.class);
  }

  @Test
  void test_type_annotation_with_any_type() {
    assertThat(lastExpression(
      "from typing import Any",
      "def f(x: Any):",
      "  x").type()).isEqualTo(anyType());
  }

  @Test
  void test_union_type_annotation_with_builtins() {
    DeclaredType union = new DeclaredType(new SymbolImpl("Union", "typing.Union"), List.of(new DeclaredType(BuiltinTypes.INT), new DeclaredType(BuiltinTypes.STR)));
    assertThat(lastExpression(
      "def f(x: int | str):",
      "  x").type()).isEqualTo(union);
  }

  @Test
  void test_union_type_annotation_inside_generics() {
    DeclaredType union = new DeclaredType(new SymbolImpl("Union", "typing.Union"), List.of(new DeclaredType(BuiltinTypes.INT), new DeclaredType(BuiltinTypes.STR)));
    DeclaredType listOfUnion = new DeclaredType(new SymbolImpl(BuiltinTypes.LIST, BuiltinTypes.LIST), List.of(union));
    assertThat(lastExpression(
      "def f(x: list[int | str]):",
      "  x").type()).isEqualTo(listOfUnion);
  }

  @Test
  void test_union_type_annotation_with_unknown_types() {
    assertThat(lastExpression(
      "def f(x: int | A):",
      "  x").type()).isEqualTo(anyType());

    assertThat(lastExpression(
      "def f(x: B | int):",
      "  x").type()).isEqualTo(anyType());
  }


  @Test
  void test_union_type_annotation_with_class() {
    assertThat(((DeclaredType) lastExpression(
      "class A:",
      "    pass",
      "def f(x: int | A):",
      "  x").type()).alternativeTypeSymbols()).extracting(Symbol::fullyQualifiedName).containsExactlyInAnyOrder("mod1.A", "int");
  }

  @Test
  void test_optional_type_annotation() {
    DeclaredType optional = new DeclaredType(new SymbolImpl("Optional", "typing.Optional"), List.of(new DeclaredType(BuiltinTypes.INT)));
    assertThat(lastExpression(
      "def f(x: int | None):",
      "  x").type()).isEqualTo(optional);

    assertThat(lastExpression(
      "def f(x: None | int):",
      "  x").type()).isEqualTo(optional);
  }
  @Test
  void test_typeSymbol() {
    ClassSymbol str = typeShedClass("str");
    assertThat(InferredTypes.typeSymbols(STR)).containsExactly(str);

    ClassSymbol a = new ClassSymbolImpl("A", "mod.A");
    assertThat(InferredTypes.typeSymbols(new RuntimeType(a))).containsExactly(a);

    assertThat(InferredTypes.typeSymbols(or(STR, INT))).containsExactlyInAnyOrder(str, typeShedClass("int"));
    assertThat(InferredTypes.typeSymbols(InferredTypes.anyType())).isEmpty();

    assertThat(InferredTypes.typeSymbols(declaredType(str))).containsExactly(str);
    assertThat(InferredTypes.typeSymbols(declaredType(new SymbolImpl("foo", "foo.bar")))).isEmpty();
  }

  @Test
  void test_typeName() {
    assertThat(InferredTypes.typeName(STR)).isEqualTo("str");

    ClassSymbol a = new ClassSymbolImpl("A", "mod.A");
    assertThat(InferredTypes.typeName(new RuntimeType(a))).isEqualTo("A");

    assertThat(InferredTypes.typeName(or(STR, INT))).isNull();
    assertThat(InferredTypes.typeName(InferredTypes.anyType())).isNull();
  }

  @Test
  void test_fullyQualifiedTypeName() {
    assertThat(InferredTypes.fullyQualifiedTypeName(STR)).isEqualTo("str");

    ClassSymbol a = new ClassSymbolImpl("A", "mod.A");
    assertThat(InferredTypes.fullyQualifiedTypeName(new RuntimeType(a))).isEqualTo("mod.A");

    assertThat(InferredTypes.fullyQualifiedTypeName(new DeclaredType(a))).isEqualTo("mod.A");

    assertThat(InferredTypes.fullyQualifiedTypeName(or(STR, INT))).isNull();
    assertThat(InferredTypes.fullyQualifiedTypeName(InferredTypes.anyType())).isNull();
  }

  @Test
  void test_typeLocation() {
    assertThat(InferredTypes.typeClassLocation(STR)).isNull();

    LocationInFile locationA = new LocationInFile("foo.py", 1, 1, 1, 1);
    RuntimeType aType = new RuntimeType(new ClassSymbolImpl("A", "mod.A", locationA));
    assertThat(InferredTypes.typeClassLocation(aType)).isEqualTo(locationA);

    LocationInFile locationB = new LocationInFile("foo.py", 1, 2, 1, 2);
    RuntimeType bType = new RuntimeType(new ClassSymbolImpl("B", "mod.B", locationB));
    assertThat(InferredTypes.typeClassLocation(or(aType, bType))).isNull();
    assertThat(InferredTypes.typeClassLocation(InferredTypes.anyType())).isNull();
  }

  @Test
  void test_isDeclaredTypeWithTypeClass() {
    assertThat(isDeclaredTypeWithTypeClass(DECL_INT, "int")).isTrue();
    assertThat(isDeclaredTypeWithTypeClass(INT, "int")).isFalse();
    assertThat(isDeclaredTypeWithTypeClass(lastExpression(
      "from typing import List",
      "def f(p: List[int]): p"
    ).type(), "list")).isTrue();
  }

  @Test
  void test_containsDeclaredType() {
    assertThat(containsDeclaredType(INT)).isFalse();
    DeclaredType declaredType = new DeclaredType(new SymbolImpl("foo", "foo"));
    assertThat(containsDeclaredType(declaredType)).isTrue();
    assertThat(containsDeclaredType(or(declaredType, INT))).isTrue();
    assertThat(containsDeclaredType(anyType())).isFalse();
  }

  @Test
  void test_type_from_protobuf() throws TextFormat.ParseException {
    assertThat(protobufType("")).isEqualTo(anyType());
    assertThat(protobufType(
      "pretty_printed_name: \"None\"\n" +
      "kind: NONE\n")).isEqualTo(NONE);
    assertThat(protobufType("kind: TYPED_DICT")).isEqualTo(DICT);
    assertThat(protobufType("kind: TUPLE")).isEqualTo(TUPLE);
    assertThat(protobufType("kind: TYPE")).isEqualTo(TYPE);
    assertThat(protobufType(
      "pretty_printed_name: \"builtins.str\"\n" +
      "fully_qualified_name: \"builtins.str\"\n")).isEqualTo(STR);
    assertThat(protobufType(
      "kind: TYPE_ALIAS\n" +
      "args {\n" +
      "  kind: UNION\n" +
      "  args {\n" +
      "    fully_qualified_name: \"builtins.str\"\n" +
      "  }\n" +
      "  args {\n" +
      "    fully_qualified_name: \"builtins.int\"\n" +
      "  }\n" +
      "}\n" +
      "fully_qualified_name: \"mod.t\""
      ))
      .isEqualTo(InferredTypes.or(STR, INT));
    assertThat(protobufType(
      "kind: INSTANCE\n" +
      "fully_qualified_name: \"typing._SpecialForm\""
      )).isEqualTo(anyType());
    assertThat(protobufType("kind: CALLABLE")).isEqualTo(anyType());
  }

  @Test
  void test_builtin_category() {
    assertThat(getBuiltinCategory(STR)).isEqualTo(BuiltinTypes.STR);
    assertThat(getBuiltinCategory(INT)).isEqualTo("number");
    assertThat(getBuiltinCategory(COMPLEX)).isEqualTo("number");
    assertThat(getBuiltinCategory(FLOAT)).isEqualTo("number");
    assertThat(getBuiltinCategory(DICT)).isEqualTo(BuiltinTypes.DICT);
    assertThat(getBuiltinCategory(LIST)).isEqualTo(BuiltinTypes.LIST);
    assertThat(getBuiltinCategory(SET)).isEqualTo(BuiltinTypes.SET);
    assertThat(getBuiltinCategory(TUPLE)).isEqualTo(BuiltinTypes.TUPLE);
    assertThat(getBuiltinCategory(anyType())).isNull();
    assertThat(getBuiltinsTypeCategory()).isNotNull();
    assertThat(getBuiltinsTypeCategory()).isNotEmpty();
  }

  @Test
  void special_form_should_be_treated_as_any() {
    assertThat(lastExpression(
      "import collections.abc as collections_abc",
      "collections_abc.Callable" // has type typing._SpecialForm
    ).type()).isEqualTo(anyType());
  }

  @Test
  void runtime_type_symbol() {
    assertThat(INT.runtimeTypeSymbol()).isEqualTo(typeShedClass("int"));
    assertThat(InferredTypes.or(INT, STR).runtimeTypeSymbol()).isNull();
    assertThat(DECL_INT.runtimeTypeSymbol()).isNull();
    assertThat(anyType().runtimeTypeSymbol()).isNull();
  }

  @Test
  void adding_a_negative_number() {
    assertThat(lastExpressionInFunction(
      "x = -1",
      "x + 1"
    ).type()).isEqualTo(INT);
  }

  private static InferredType protobufType(String protobuf) throws TextFormat.ParseException {
    SymbolsProtos.Type.Builder builder = SymbolsProtos.Type.newBuilder();
    TextFormat.merge(protobuf, builder);
    return fromTypeshedProtobuf(builder.build());
  }

  private TypeAnnotation typeAnnotation(String... code) {
    return PythonTestUtils.getLastDescendant(PythonTestUtils.parse(code), tree -> tree.is(Tree.Kind.VARIABLE_TYPE_ANNOTATION));
  }

  private DeclaredType declaredType(Symbol symbol) {
    return new DeclaredType(symbol);
  }
}
