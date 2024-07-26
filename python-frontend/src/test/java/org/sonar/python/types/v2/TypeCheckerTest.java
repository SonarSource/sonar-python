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
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.tree.TreeUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.types.v2.ClassTypeTest.classType;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;


class TypeCheckerTest {

  TypeChecker typeChecker = new TypeChecker(PROJECT_LEVEL_TYPE_TABLE);

  @Test
  void isBuiltinWithNameTest() {
    FileInput fileInput = parseAndInferTypes("42");
    NumericLiteral intLiteral = (NumericLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NUMERIC_LITERAL)).get();
    ObjectType intLiteralType = (ObjectType) intLiteral.typeV2();
    assertThat(intLiteralType.unwrappedType()).isEqualTo(INT_TYPE);

    assertThat(typeChecker.typeCheckBuilder().isBuiltinWithName("int").check(intLiteralType)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().isBuiltinWithName("str").check(intLiteralType)).isEqualTo(TriBool.FALSE);
    assertThat(typeChecker.typeCheckBuilder().isBuiltinWithName("unknown").check(intLiteralType)).isEqualTo(TriBool.UNKNOWN);

    fileInput = parseAndInferTypes("foo()");
    CallExpression callExpression = (CallExpression) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.CALL_EXPR)).get();
    PythonType callExpressionType = callExpression.typeV2();
    assertThat(callExpressionType).isEqualTo(PythonType.UNKNOWN);
    assertThat(typeChecker.typeCheckBuilder().isBuiltinWithName("int").check(callExpressionType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeChecker.typeCheckBuilder().isBuiltinWithName("unknown").check(callExpressionType)).isEqualTo(TriBool.UNKNOWN);

    assertThat(typeChecker.typeCheckBuilder().isBuiltinWithName("int").check(INT_TYPE)).isEqualTo(TriBool.TRUE);
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

    assertThat(typeChecker.typeCheckBuilder().hasMember("foo").check(objectType)).isEqualTo(TriBool.FALSE);
    assertThat(typeChecker.typeCheckBuilder().instancesHaveMember("foo").check(objectType)).isEqualTo(TriBool.FALSE);

    fileInput = parseAndInferTypes("""
      class A:
        def foo(self): ...
      a = A()
      a
      """
    );
    objectType = (ObjectType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(typeChecker.typeCheckBuilder().hasMember("foo").check(objectType)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().instancesHaveMember("foo").check(objectType)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void classTypeHasMemberTest() {
    ClassType classType = classType("class C: ...");

    assertThat(typeChecker.typeCheckBuilder().hasMember("__call__").check(classType)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().hasMember("unknown").check(classType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeChecker.typeCheckBuilder().instancesHaveMember("__call__").check(classType)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void typeSourceCheckTest() {
    var fileInput = parseAndInferTypes("""
      def foo(x: int):
        y = 10
        x
        y
      """
    );
    var functionBodyStatements = ((FunctionDef) fileInput.statements().statements().get(0)).body().statements();
    var xName = ((ExpressionStatement) functionBodyStatements.get(1)).expressions().get(0);
    var yName = ((ExpressionStatement) functionBodyStatements.get(2)).expressions().get(0);
    var xType = xName.typeV2();
    var yType = yName.typeV2();

    assertThat(typeChecker.typeCheckBuilder().isTypeHintTypeSource().check(xType)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().isExactTypeSource().check(xType)).isEqualTo(TriBool.FALSE);

    assertThat(typeChecker.typeCheckBuilder().isTypeHintTypeSource().check(yType)).isEqualTo(TriBool.FALSE);
    assertThat(typeChecker.typeCheckBuilder().isExactTypeSource().check(yType)).isEqualTo(TriBool.TRUE);
  }
}
