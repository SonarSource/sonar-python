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
package org.sonar.python.types.v2;

import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NoneExpression;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.SymbolUtils;
import org.sonar.python.tree.TreeUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;


class ObjectTypeTest {

  @Test
  void simpleObject() {
    PythonFile pythonFile = PythonTestUtils.pythonFile("");
    FileInput fileInput = parseAndInferTypes(pythonFile, """
      class A: ...
      a = A()
      a
      """
    );
    ClassType classType = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    ObjectType objectType = (ObjectType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(objectType.unwrappedType()).isEqualTo(classType);

    assertThat(objectType.displayName()).contains("A");
    assertThat(objectType.isCompatibleWith(classType)).isTrue();
    assertThat(objectType.hasMember("foo")).isEqualTo(TriBool.FALSE);
    String fileId = SymbolUtils.pathOf(pythonFile).toString();
    assertThat(objectType.definitionLocation()).contains(new LocationInFile(fileId, 1, 6, 1, 7));
    assertThat(TypeUtils.resolved(objectType)).isEqualTo(objectType);
  }

  @Test
  void simpleObjectWithMember() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        def foo(self): ...
      a = A()
      a
      """
    );
    ClassType classType = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    ObjectType objectType = (ObjectType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();

    assertThat(objectType.displayName()).contains("A");
    assertThat(objectType.isCompatibleWith(classType)).isTrue();
    assertThat(objectType.hasMember("foo")).isEqualTo(TriBool.TRUE);
  }

  @Test
  void reassignedObject() {
    FileInput fileInput = parseAndInferTypes("""
      class A:
        def foo(self): ...
      a = A()
      a = B()
      a
      """
    );
    ClassType classType = (ClassType) ((ClassDef) fileInput.statements().statements().get(0)).name().typeV2();
    PythonType aType = ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0).typeV2();

    assertThat(aType).isEqualTo(PythonType.UNKNOWN);
    assertThat(aType.isCompatibleWith(classType)).isTrue();
    assertThat(aType.hasMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void literalTypes() {
    FileInput fileInput = parseAndInferTypes("\"hello\"");
    StringLiteral stringLiteral = (StringLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.STRING_LITERAL)).get();
    ObjectType stringLiteralType = (ObjectType) stringLiteral.typeV2();
    assertThat(stringLiteralType.displayName()).contains("str");

    fileInput = parseAndInferTypes("(1, 2, 3)");
    Tuple intTuple = (Tuple) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.TUPLE)).get();
    ObjectType intTupleType = (ObjectType) intTuple.typeV2();
    assertThat(intTupleType.displayName()).contains("tuple");
    assertThat(intTupleType.attributes()).extracting(PythonType::displayName).extracting(Optional::get).containsExactly("int");

    fileInput = parseAndInferTypes("(1, \"hello\")");
    Tuple intStrTuple = (Tuple) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.TUPLE)).get();
    ObjectType intStrTupleType = (ObjectType) intStrTuple.typeV2();
    assertThat(intStrTupleType.displayName()).contains("tuple");
    assertThat(intStrTupleType.attributes()).isEmpty();

    fileInput = parseAndInferTypes("(foo(),)");
    Tuple unknownTuple = (Tuple) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.TUPLE)).get();
    ObjectType unknownTupleType = (ObjectType) unknownTuple.typeV2();
    assertThat(unknownTupleType.displayName()).contains("tuple");
    assertThat(unknownTupleType.attributes()).isEmpty();

    fileInput = parseAndInferTypes("{1, 2, 3}");
    SetLiteral setLiteral = (SetLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.SET_LITERAL)).get();
    ObjectType setLiteralType = (ObjectType) setLiteral.typeV2();
    assertThat(setLiteralType.displayName()).contains("set");

    fileInput = parseAndInferTypes("{\"my_key\": 42}");
    DictionaryLiteral dictionaryLiteral = (DictionaryLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.DICTIONARY_LITERAL)).get();
    ObjectType dictionaryLiteralType = (ObjectType) dictionaryLiteral.typeV2();
    assertThat(dictionaryLiteralType.displayName()).contains("dict");

    fileInput = parseAndInferTypes("None");
    NoneExpression noneExpression = (NoneExpression) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NONE)).get();
    ObjectType noneExpressionType = (ObjectType) noneExpression.typeV2();
    assertThat(noneExpressionType.displayName()).contains("NoneType");

    fileInput = parseAndInferTypes("unknown");
    Name name = (Name) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NAME)).get();
    UnknownType nameType = (UnknownType) name.typeV2();
    assertThat(nameType.displayName()).isEmpty();
  }

  @Test
  void numericLiteralTypes() {
    FileInput fileInput = parseAndInferTypes("42");
    NumericLiteral intLiteral = (NumericLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NUMERIC_LITERAL)).get();
    ObjectType intLiteralType = (ObjectType) intLiteral.typeV2();
    assertThat(intLiteralType.displayName()).contains("int");

    fileInput = parseAndInferTypes("42.5");
    NumericLiteral floatLiteral = (NumericLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NUMERIC_LITERAL)).get();
    ObjectType floatLiteralType = (ObjectType) floatLiteral.typeV2();
    assertThat(floatLiteralType.displayName()).contains("float");

    fileInput = parseAndInferTypes("4e2");
    NumericLiteral exponentLiteral = (NumericLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NUMERIC_LITERAL)).get();
    ObjectType exponentLiteralType = (ObjectType) exponentLiteral.typeV2();
    assertThat(exponentLiteralType.displayName()).contains("float");

    fileInput = parseAndInferTypes("42j");
    NumericLiteral complexLiteral = (NumericLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NUMERIC_LITERAL)).get();
    ObjectType complexLiteralType = (ObjectType) complexLiteral.typeV2();
    assertThat(complexLiteralType.displayName()).contains("complex");
  }

  @Test
  void function_is_callable() {
      ClassType classType = new ClassType("function", "builtins.function");
      assertThat(classType.instancesHaveMember("__call__")).isEqualTo(TriBool.TRUE);
      assertThat(classType.instancesHaveMember("other")).isEqualTo(TriBool.FALSE);
  }

  @Test
  void objectType_of_unknown() {
    // TODO SONARPY-1875: Ensure this is the behavior we want (do we even want it possible to have object of unknown? Maybe replace with UnionType when implemented
    ObjectType objectType = new ObjectType(PythonType.UNKNOWN, List.of(), List.of());
    assertThat(objectType.hasMember("foo")).isEqualTo(TriBool.UNKNOWN);
  }
}
