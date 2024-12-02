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
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.tree.TreeUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.types.v2.ClassTypeTest.classType;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.LIST_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.SET_TYPE;
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

  @Test
  void isInstanceOfSimpleTest() {
    var listObject = new ObjectType(LIST_TYPE);
    var setObject = new ObjectType(SET_TYPE);
    var listOrSetObject = new ObjectType(UnionType.or(LIST_TYPE, SET_TYPE));

    var checker = typeChecker.typeCheckBuilder().isInstanceOf("list");
    assertThat(checker.check(listObject)).isEqualTo(TriBool.TRUE);
    assertThat(checker.check(setObject)).isEqualTo(TriBool.FALSE);
    assertThat(checker.check(listOrSetObject)).isEqualTo(TriBool.UNKNOWN);
    assertThat(checker.check(LIST_TYPE)).isEqualTo(TriBool.UNKNOWN);
    assertThat(checker.check(SET_TYPE)).isEqualTo(TriBool.UNKNOWN);

    var unknownTypeChecker = typeChecker.typeCheckBuilder().isInstanceOf("typing.something");
    assertThat(unknownTypeChecker.check(listObject)).isEqualTo(TriBool.UNKNOWN);
    assertThat(unknownTypeChecker.check(setObject)).isEqualTo(TriBool.UNKNOWN);
    assertThat(unknownTypeChecker.check(listOrSetObject)).isEqualTo(TriBool.UNKNOWN);
    assertThat(unknownTypeChecker.check(LIST_TYPE)).isEqualTo(TriBool.UNKNOWN);
    assertThat(unknownTypeChecker.check(SET_TYPE)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void isInstanceOfObjectOfUnionTypeTest() {
    var fileInput = parseAndInferTypes("""
      import typing
      
      class A(typing.Iterator): ...
      class B(typing.Iterator): ...
      class C: ...
      class D: ...
      
      def foo(p):
        if p == 1:
          xt = A
          yt = A
          zt = C
        elif p == 2:
          xt = B
          yt = B
          zt = C
        else:
          xt = B
          yt = C
          zt = D
        x = xt()
        y = yt()
        z = zt()
        x
        y
        z
      """
    );

    var statements = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::body)
      .map(StatementList::statements)
      .orElseGet(List::of);
    var xType = ((ExpressionStatement) statements.get(statements.size() - 3)).expressions().get(0).typeV2();
    var yType = ((ExpressionStatement) statements.get(statements.size() - 2)).expressions().get(0).typeV2();
    var zType = ((ExpressionStatement) statements.get(statements.size() - 1)).expressions().get(0).typeV2();

    var checker = typeChecker.typeCheckBuilder().isInstanceOf("typing.Iterator");
    assertThat(checker.check(xType)).isEqualTo(TriBool.TRUE);
    assertThat(checker.check(yType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(checker.check(zType)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void isInstanceOfUnionTypeTest() {
    var fileInput = parseAndInferTypes("""
      import typing
      
      class A(typing.Iterator): ...
      class B(typing.Iterator): ...
      class C: ...
      class D: ...
      
      def foo(p):
        if p:
          x = A()
          y = A()
          z = C()
        else:
          x = B()
          y = C()
          z = D()
        x
        y
        z
      """
    );

    var statements = TreeUtils.firstChild(fileInput, FunctionDef.class::isInstance)
      .map(FunctionDef.class::cast)
      .map(FunctionDef::body)
      .map(StatementList::statements)
      .orElseGet(List::of);
    var xType = ((ExpressionStatement) statements.get(statements.size() - 3)).expressions().get(0).typeV2();
    var yType = ((ExpressionStatement) statements.get(statements.size() - 2)).expressions().get(0).typeV2();
    var zType = ((ExpressionStatement) statements.get(statements.size() - 1)).expressions().get(0).typeV2();

    var checker = typeChecker.typeCheckBuilder().isInstanceOf("typing.Iterator");
    assertThat(checker.check(xType)).isEqualTo(TriBool.TRUE);
    assertThat(checker.check(yType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(checker.check(zType)).isEqualTo(TriBool.FALSE);
  }


  @Test
  void isTypeWithNameStubNamesTest() {
    FileInput fileInput = parseAndInferTypes("42");
    NumericLiteral intLiteral = (NumericLiteral) TreeUtils.firstChild(fileInput, t -> t.is(Tree.Kind.NUMERIC_LITERAL)).get();
    ObjectType intLiteralType = (ObjectType) intLiteral.typeV2();
    assertThat(intLiteralType.unwrappedType()).isEqualTo(INT_TYPE);

    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("int").check(intLiteralType)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("str").check(intLiteralType)).isEqualTo(TriBool.FALSE);
    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("unknown").check(intLiteralType)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeChecker.typeCheckBuilder().isTypeWithName("int").check(intLiteralType)).isEqualTo(TriBool.FALSE);
    assertThat(typeChecker.typeCheckBuilder().isTypeWithName("int").check(intLiteralType.unwrappedType())).isEqualTo(TriBool.TRUE);

    fileInput = parseAndInferTypes("round(42.42)");
    var roundType = ((CallExpression) ((ExpressionStatement) fileInput.statements().statements().get(0)).expressions().get(0)).callee().typeV2();
    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("round").check(roundType)).isEqualTo(TriBool.TRUE);

    fileInput = parseAndInferTypes(
      """
        from flask import Response
        Response()
        """
    );
    var responseType = ((CallExpression) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0)).callee().typeV2();
    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("flask.Response").check(responseType)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("flask.wrappers.Response").check(responseType)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("flask.app.Response").check(responseType)).isEqualTo(TriBool.UNKNOWN);

    FunctionType maxCookieSize = (FunctionType) responseType.resolveMember("max_cookie_size").get();
    assertThat(maxCookieSize.fullyQualifiedName()).isEqualTo("flask.wrappers.Response.max_cookie_size");
    assertThat(typeChecker.typeCheckBuilder().isTypeWithName("flask.wrappers.Response.max_cookie_size").check(maxCookieSize)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().isTypeWithName("flask.Response.max_cookie_size").check(maxCookieSize)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().isTypeWithName("flask.Response.unknown.max_cookie_size").check(maxCookieSize)).isEqualTo(TriBool.UNKNOWN);
    assertThat(typeChecker.typeCheckBuilder().isTypeWithName("flask.Response.autocorrect_location_header").check(maxCookieSize)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void isTypeWithNameProjectNamesTest() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();

    FileInput tree = parseWithoutSymbols(
      """
      class A: pass
      class B: pass
      """
    );
    PythonFile pythonFile = PythonTestUtils.pythonFile("mod.py");
    projectLevelSymbolTable.addModule(tree, "my_package", pythonFile);
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    TypeChecker localTypeChecker = new TypeChecker(projectLevelTypeTable);

    FileInput initTree = parseWithoutSymbols("");
    PythonFile initFile = PythonTestUtils.pythonFile("__init__.py");
    projectLevelSymbolTable.addModule(initTree, "my_package", initFile);

    var fileInput = parseAndInferTypes(projectLevelTypeTable, pythonFile, """
      from my_package.mod import A
      A
      """
    );
    var aType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(localTypeChecker.typeCheckBuilder().isTypeOrInstanceWithName("my_package.mod.A").check(aType)).isEqualTo(TriBool.TRUE);
    assertThat(localTypeChecker.typeCheckBuilder().isTypeOrInstanceWithName("my_package.mod.B").check(aType)).isEqualTo(TriBool.FALSE);
    assertThat(localTypeChecker.typeCheckBuilder().isTypeOrInstanceWithName("my_package.unknown.A").check(aType)).isEqualTo(TriBool.UNKNOWN);
  }


  @Test
  void testIsInstance() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    var builder = new TypeCheckBuilder(table).isInstance();

    var intType = table.getType("int");
    var floatType = table.getType("float");

    var unionOfAllObjects = UnionType.or(new ObjectType(intType), new ObjectType(floatType));
    var unionOfSomeObjects = UnionType.or(new ObjectType(intType), floatType);
    var unionOfNoObjects = UnionType.or(intType, floatType);

    assertThat(builder.check(new ObjectType(intType)))
      .isEqualTo(TriBool.TRUE);

    assertThat(builder.check(intType))
      .isEqualTo(TriBool.FALSE);

    assertThat(builder.check(unionOfAllObjects))
      .isEqualTo(TriBool.TRUE);
    assertThat(builder.check(unionOfSomeObjects))
      .isEqualTo(TriBool.FALSE);
    assertThat(builder.check(unionOfNoObjects))
      .isEqualTo(TriBool.FALSE);
  }


  @Test
  void isGenericTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);
    var builder = new TypeCheckBuilder(table).isGeneric();

    var typingGenericType = table.getType("typing.Generic");
    var typingUnionType = table.getType("typing.Union");

    assertThat(builder.check(typingGenericType)).isEqualTo(TriBool.TRUE);
    assertThat(builder.check(typingUnionType)).isEqualTo(TriBool.UNKNOWN);

    var unresolvedTypingGeneric = new UnknownType.UnresolvedImportType("typing.Generic");
    var unresolvedTypingUnion = new UnknownType.UnresolvedImportType("typing.Union");
    assertThat(builder.check(unresolvedTypingGeneric)).isEqualTo(TriBool.TRUE);
    assertThat(builder.check(unresolvedTypingUnion)).isEqualTo(TriBool.UNKNOWN);

    var intType = table.getType("int");
    assertThat(builder.check(intType)).isEqualTo(TriBool.UNKNOWN);

    var basicClassType = new ClassType("MyClass", "mod.MyClass");
    assertThat(builder.check(basicClassType)).isEqualTo(TriBool.UNKNOWN);

    var genericClassType = new ClassType("MyClass", "mod.MyClass", Set.of(), List.of(), List.of(), List.of(), false, true, null);
    assertThat(builder.check(genericClassType)).isEqualTo(TriBool.TRUE);
  }
}
