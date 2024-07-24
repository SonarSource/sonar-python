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

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.tree.TreeUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.ClassTypeTest.classType;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;


class TypeCheckerContextTest {

  TypeCheckerContext typeCheckerContext = new TypeCheckerContext(PROJECT_LEVEL_TYPE_TABLE);

  @Test
  void isBuiltinWithNameTest() {
    FileInput fileInput = parseAndInferTypes("42");
    NumericLiteral intLiteral = (NumericLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NUMERIC_LITERAL)).get();
    ObjectType intLiteralType = (ObjectType) intLiteral.typeV2();
    assertThat(intLiteralType.unwrappedType()).isEqualTo(INT_TYPE);

    assertThat(typeCheckerContext.typeChecker().isBuiltinWithName("int").check(intLiteralType)).isEqualTo(TriBool.TRUE);
    assertThat(typeCheckerContext.typeChecker().isBuiltinWithName("str").check(intLiteralType)).isEqualTo(TriBool.FALSE);
    assertThat(typeCheckerContext.typeChecker().isBuiltinWithName("unknown").check(intLiteralType)).isEqualTo(TriBool.UNKNOWN);

    fileInput = parseAndInferTypes("foo()");
    CallExpression callExpression = (CallExpression) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.CALL_EXPR)).get();
    PythonType callExpressionType = callExpression.typeV2();
    assertThat(callExpressionType).isEqualTo(PythonType.UNKNOWN);
    assertThat(typeCheckerContext.typeChecker().isBuiltinWithName("int").check(callExpressionType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeCheckerContext.typeChecker().isBuiltinWithName("unknown").check(callExpressionType)).isEqualTo(TriBool.UNKNOWN);

    assertThat(typeCheckerContext.typeChecker().isBuiltinWithName("int").check(INT_TYPE)).isEqualTo(TriBool.TRUE);
  }


  @Test
  void objectTypeHasMemberTest() {
    PythonFile pythonFile = PythonTestUtils.pythonFile("");
    FileInput fileInput = parseAndInferTypes(pythonFile, """
      class A: ...
      a = A()
      a
      """
    );
    ObjectType objectType = (ObjectType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();

    assertThat(typeCheckerContext.typeChecker().hasMember("foo").check(objectType)).isEqualTo(TriBool.FALSE);
    assertThat(typeCheckerContext.typeChecker().instancesHaveMember("foo").check(objectType)).isEqualTo(TriBool.FALSE);

    fileInput = parseAndInferTypes("""
      class A:
        def foo(self): ...
      a = A()
      a
      """
    );
    objectType = (ObjectType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(typeCheckerContext.typeChecker().hasMember("foo").check(objectType)).isEqualTo(TriBool.TRUE);
    assertThat(typeCheckerContext.typeChecker().instancesHaveMember("foo").check(objectType)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void classTypeHasMemberTest() {
    ClassType classType = classType("class C: ...");

    assertThat(typeCheckerContext.typeChecker().hasMember("__call__").check(classType)).isEqualTo(TriBool.TRUE);
    assertThat(typeCheckerContext.typeChecker().hasMember("unknown").check(classType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeCheckerContext.typeChecker().instancesHaveMember("__call__").check(classType)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void orTest() {
    FileInput fileInput = parseAndInferTypes("42");
    NumericLiteral intLiteral = (NumericLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NUMERIC_LITERAL)).get();
    ObjectType intLiteralType = (ObjectType) intLiteral.typeV2();
    assertThat(intLiteralType.unwrappedType()).isEqualTo(INT_TYPE);

    assertThat(typeCheckerContext.typeChecker().isBuiltinWithName("str").or(
      typeCheckerContext.typeChecker().isBuiltinWithName("int")
    ).check(intLiteralType)).isEqualTo(TriBool.TRUE);

    assertThat(typeCheckerContext.typeChecker().isBuiltinWithName("int").or(
      typeCheckerContext.typeChecker().isBuiltinWithName("str")
    ).check(intLiteralType)).isEqualTo(TriBool.TRUE);

    assertThat(typeCheckerContext.typeChecker().isBuiltinWithName("int").or(
      typeCheckerContext.typeChecker().hasMember("__abs__")
    ).check(intLiteralType)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void andTest() {
    FileInput fileInput = parseAndInferTypes("42");
    NumericLiteral intLiteral = (NumericLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NUMERIC_LITERAL)).get();
    ObjectType intLiteralType = (ObjectType) intLiteral.typeV2();

    TypeChecker isIntAndStr = typeCheckerContext.typeChecker().isBuiltinWithName("int").isBuiltinWithName("str");
    TypeChecker isStrAndInt = typeCheckerContext.typeChecker().isBuiltinWithName("str").isBuiltinWithName("int");
    TypeChecker isIntAndHasMemberAbs = typeCheckerContext.typeChecker().isBuiltinWithName("int").hasMember("__abs__");
    TypeChecker hasMemberFloatAndEq = typeCheckerContext.typeChecker().hasMember("__eq__").hasMember("__float__");
    TypeChecker isUnknownAndHasMemberFloat = typeCheckerContext.typeChecker().isBuiltinWithName("UNKNOWN_BUILTIN_NAME").hasMember("__float__");

    assertThat(isIntAndStr.check(intLiteralType)).isEqualTo(TriBool.FALSE);
    assertThat(isStrAndInt.check(intLiteralType)).isEqualTo(TriBool.FALSE);
    assertThat(isIntAndHasMemberAbs.check(intLiteralType)).isEqualTo(TriBool.TRUE);
    assertThat(isUnknownAndHasMemberFloat.check(intLiteralType)).isEqualTo(TriBool.UNKNOWN);

    assertThat(isIntAndHasMemberAbs.and(hasMemberFloatAndEq).check(intLiteralType)).isEqualTo(TriBool.TRUE);
    assertThat(isIntAndHasMemberAbs.and(isUnknownAndHasMemberFloat).check(intLiteralType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isUnknownAndHasMemberFloat.and(isIntAndHasMemberAbs).check(intLiteralType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(isIntAndStr.and(isStrAndInt).check(intLiteralType)).isEqualTo(TriBool.FALSE);

    TypeChecker trueOr1 = isStrAndInt.or(isIntAndHasMemberAbs);
    TypeChecker trueOr2 = isIntAndHasMemberAbs.or(isStrAndInt);
    TypeChecker trueOr3 = isIntAndHasMemberAbs.or(isUnknownAndHasMemberFloat);
    TypeChecker trueOr4 = isUnknownAndHasMemberFloat.or(isIntAndHasMemberAbs);

    assertThat(trueOr1.check(intLiteralType)).isEqualTo(TriBool.TRUE);
    assertThat(trueOr2.check(intLiteralType)).isEqualTo(TriBool.TRUE);
    assertThat(trueOr3.check(intLiteralType)).isEqualTo(TriBool.TRUE);
    assertThat(trueOr4.check(intLiteralType)).isEqualTo(TriBool.TRUE);
  }
}
