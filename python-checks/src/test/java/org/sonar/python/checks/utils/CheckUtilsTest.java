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
package org.sonar.python.checks.utils;

import com.sonar.sslr.api.AstNode;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Scanner;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.ArgListImpl;
import org.sonar.python.tree.PythonTreeMaker;
import org.sonar.python.tree.TreeUtils;

import static org.assertj.core.api.Assertions.assertThat;

class CheckUtilsTest {

  @Test
  void private_constructor() throws Exception {
    Constructor constructor = CheckUtils.class.getDeclaredConstructor();
    assertThat(constructor.isAccessible()).isFalse();
    constructor.setAccessible(true);
    constructor.newInstance();
  }

  @Test
  void statement_equivalence() {
    assertThat(CheckUtils.areEquivalent(parse("x = x + 1"), parse("x = x + 1"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x = x + 1"), parse("x = x + 1"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x = x + 1"), parse("x = x + 1"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x = x + 1"), parse("x = x + 2"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo()"), parse("foo()"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("foo()"), parse("foo"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo"), parse("foo()"))).isFalse();
  }

  @Test
  void comparison_equivalence() {
    assertThat(CheckUtils.areEquivalent(parse("foo is None"), parse("foo is not None"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("x < 2"), parse("x > 2"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo is None"), parse("foo is  None"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x < 1"), parse("x < 1"))).isTrue();
  }

  @Test
  void tree_equivalence() {
    assertThat(CheckUtils.areEquivalent(new ArgListImpl(Collections.emptyList(), Collections.emptyList()),
      new ArgListImpl(Collections.emptyList(), Collections.emptyList()))).isTrue();
  }

  @Test
  void null_equivalence() {
    assertThat(CheckUtils.areEquivalent(null, null)).isTrue();
    assertThat(CheckUtils.areEquivalent(null, parse("class clazz(): \n pass"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("class clazz(): \n pass"), null)).isFalse();
  }

  @Test
  void statement_list_equivalence() {
    assertThat(CheckUtils.areEquivalent(parse("foo()\nbar()"), parse("foo()\nbar()"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("foo()\n  "), parse("foo()\n"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("foo()\n"), parse("foo()\n  "))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("foo()"), parse("foo()\nfoo()"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo()\nfoo()"), parse("foo()"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo()\nbar()"), parse("foo()\nbar"))).isFalse();
  }

  @Test
  void lambda_equivalence() {
    assertThat(CheckUtils.areEquivalent(parse("x = lambda a : a + 10"), parse("x = lambda a : a + 10"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x = lambda a : a + 10"), parse("x = lambda a : a + 5"))).isFalse();
  }

  @Test
  void no_parent_class() {
    FileInput file = (FileInput) parse("" +
      "def f():\n" +
      "    pass\n");
    FunctionDef f = descendantFunction(file, "f");
    assertThat(f).isNotNull();
    assertThat(CheckUtils.getParentClassDef(f)).isNull();
  }

  @Test
  void parent_class() {
    FileInput file = (FileInput) parse("" +
      "class A:\n" +
      "    def f():\n" +
      "        def g():\n" +
      "            pass\n" +
      "        pass\n" +
      "\n" +
      "    if x:\n" +
      "        def h():\n" +
      "            pass\n");
    FunctionDef f = descendantFunction(file, "f");
    FunctionDef g = descendantFunction(file, "g");
    FunctionDef h = descendantFunction(file, "h");
    assertThat(f).isNotNull();
    assertThat(g).isNotNull();
    assertThat(h).isNotNull();

    ClassDef parent = CheckUtils.getParentClassDef(f);
    assertThat(parent).isNotNull();
    assertThat(parent.name().name()).isEqualTo("A");

    parent = CheckUtils.getParentClassDef(g);
    assertThat(parent).isNull();

    parent = CheckUtils.getParentClassDef(h);
    assertThat(parent).isNotNull();
    assertThat(parent.name().name()).isEqualTo("A");
  }

  @Test
  void mustBeAProtocolLikeTest() throws IOException {
    var fileInput = parseFileWithSymbols("src/test/resources/checks/checkUtils/mustBeAProtocolLikeTest.py");

    var statementList = fileInput.statements();
    assertThat(statementList).isNotNull();

    statementList
      .statements()
      .stream().filter(child -> child.is(Tree.Kind.CLASSDEF))
      .map(ClassDef.class::cast)
      .forEach(classDef -> {
        var name = classDef.name().name();
        var mustBeProtocolLike = CheckUtils.mustBeAProtocolLike(classDef);

        assertThat(mustBeProtocolLike)
          .isEqualTo(name.startsWith("ProtocolLike"));
      });
  }

  @Test
  void isAbstractTest() throws IOException {
    var fileInput = parseFile("src/test/resources/checks/checkUtils/isAbstractTest.py");

    var abstractMethodNames = List.of("standard_usage", "qualified_usage", "usage_with_other_decorator", "incorrect_calling_usage", "usage_with_unknown_other_decorator");
    for (var abstractMethodName : abstractMethodNames) {
      FunctionDef method = descendantFunction(fileInput, abstractMethodName);
      assertThat(method).isNotNull();
      assertThat(CheckUtils.isAbstract(method)).isTrue();
    }

    var nonAbstractMethodNames = List.of("standard_method", "with_other_decorator", "with_unknown_decorator");
    for (var nonAbstractMethod : nonAbstractMethodNames) {
      FunctionDef method = descendantFunction(fileInput, nonAbstractMethod);
      assertThat(method).isNotNull();
      assertThat(CheckUtils.isAbstract(method)).isFalse();
    }
  }

  @Test
  void isSelfTest() throws IOException {
    var fileInput = parseFile("src/test/resources/checks/checkUtils/isSelfTest.py");
    var functionDefs = descendantFunctions(fileInput);

    assertThat(functionDefs).hasSize(7);

    for (var functionDef : functionDefs) {
      var functionName = functionDef.name().name();
      var returnStmt = TreeUtils.firstChild(functionDef, child -> child.is(Tree.Kind.RETURN_STMT));
      assertThat(returnStmt).isNotEmpty();

      var maybeSelf = ((ReturnStatement) returnStmt.get()).expressions().get(0);

      assertThat(CheckUtils.isSelf(maybeSelf)).isEqualTo(functionName.startsWith("returnsSelf"));
    }
  }

  @Test
  void findFirstParameterSymbolTest() throws IOException {
    var fileInput = parseFileWithSymbols("src/test/resources/checks/checkUtils/findFirstParameterSymbolTest.py");
    var functionDefs = descendantFunctions(fileInput);

    assertThat(functionDefs).hasSize(8);

    for (var functionDef : functionDefs) {
      var functionName = functionDef.name().name();

      assertThat(CheckUtils.findFirstParameterSymbol(functionDef) != null)
        .isEqualTo(functionName.startsWith("hasSymbolFirst"));
    }
  }

  private static Tree parse(String content) {
    PythonParser parser = PythonParser.create();
    AstNode astNode = parser.parse(content);
    FileInput parse = new PythonTreeMaker().fileInput(astNode);
    return parse;
  }

  private static FileInput parseFile(String path) throws IOException {
    try (var sourceFile = new Scanner(new File(path)).useDelimiter("\\Z")) {
      return (FileInput) parse(sourceFile.next());
    }
  }

  private static FileInput parseFileWithSymbols(String path) throws IOException {
    return TestPythonVisitorRunner
      .createContext(new File(path))
      .rootTree();
  }

  @Nullable
  private static FunctionDef descendantFunction(Tree tree, String name) {
    if (tree.is(Tree.Kind.FUNCDEF)) {
      FunctionDef functionDef = (FunctionDef) tree;
      if (functionDef.name().name().equals(name)) {
        return functionDef;
      }
    }
    return tree.children().stream()
      .map(child -> descendantFunction(child, name))
      .filter(Objects::nonNull)
      .findFirst().orElse(null);
  }

  private static List<FunctionDef> descendantFunctions(Tree tree) {
    if (tree.is(Tree.Kind.FUNCDEF)) {
      return List.of((FunctionDef) tree);
    }
    return tree.children().stream()
      .flatMap(child -> descendantFunctions(child).stream())
      .collect(Collectors.toUnmodifiableList());
  }
}
