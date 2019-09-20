/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.checks;

import com.google.common.collect.ImmutableSet;
import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.impl.Parser;
import java.lang.reflect.Constructor;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Set;
import java.util.function.Predicate;
import java.util.function.UnaryOperator;
import org.junit.Test;
import org.sonar.python.PythonCheckAstNode;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PyArgListTreeImpl;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class CheckUtilsTest {

  private static final Parser<Grammar> PARSER = PythonParser.create(new PythonConfiguration(StandardCharsets.UTF_8));

  @Test
  public void private_constructor() throws Exception {
    Constructor constructor = CheckUtils.class.getDeclaredConstructor();
    assertThat(constructor.isAccessible()).isFalse();
    constructor.setAccessible(true);
    constructor.newInstance();
  }

  @Test
  public void equals_node() {
    PythonCheckVerifier.verify("src/test/resources/checks/utils/equalsNode.py", new EqualsNodeCheck());
  }

  private static class EqualsNodeCheck extends PythonCheckAstNode {
    @Override
    public Set<AstNodeType> subscribedKinds() {
      return ImmutableSet.copyOf(PythonGrammar.values());
    }

    @Override
    public void visitNode(AstNode astNode) {
      assertTrue(CheckUtils.equalNodes(astNode, astNode));
      AstNode parent = astNode.getParent();
      if (parent != null) {
        assertFalse(CheckUtils.equalNodes(astNode, parent));
      }
      for (AstNode child : astNode.getChildren()) {
        assertFalse(CheckUtils.equalNodes(astNode, child));
      }
    }
  }

  @Test
  public void is_method_definition() {
    PythonCheckVerifier.verify("src/test/resources/checks/utils/isMethodDefinition.py", new IsMethodDefinitionCheck());
  }

  private static class IsMethodDefinitionCheck extends PythonCheckAstNode {
    @Override
    public Set<AstNodeType> subscribedKinds() {
      return ImmutableSet.of(
        PythonGrammar.FILE_INPUT,
        PythonGrammar.CLASSDEF,
        PythonGrammar.FUNCDEF
      );
    }

    @Override
    public void visitNode(AstNode astNode) {
      if (CheckUtils.isMethodDefinition(astNode)) {
        addIssue(astNode, "method_definition");
      } else {
        addIssue(astNode, "not_method_definition");
      }
    }
  }

  @Test
  public void class_has_inheritance() {
    PythonCheckVerifier.verify("src/test/resources/checks/utils/classHasInheritance.py", new ClassHasInheritanceCheck());
  }

  private static class ClassHasInheritanceCheck extends PythonCheckAstNode {
    @Override
    public Set<AstNodeType> subscribedKinds() {
      return ImmutableSet.of(PythonGrammar.CLASSDEF);
    }

    @Override
    public void visitNode(AstNode astNode) {
      if (CheckUtils.classHasInheritance(astNode)) {
        addIssue(astNode, "has_inheritance");
      } else {
        addIssue(astNode, "no_inheritance");
      }
    }
  }

  @Test
  public void string_interpolation() {
    Predicate<String> isStringInterpolation = source -> CheckUtils.isStringInterpolation(PARSER.parse(source).getTokens().get(0));
    assertThat(isStringInterpolation.test("\"abc\"")).isFalse();
    assertThat(isStringInterpolation.test("r'abc'")).isFalse();
    assertThat(isStringInterpolation.test("f'abc'")).isTrue();
    assertThat(isStringInterpolation.test("Rf'abc'")).isTrue();
    assertThat(isStringInterpolation.test("rF'abc'")).isTrue();
    assertThat(isStringInterpolation.test("fr\"\"")).isTrue();
  }

  @Test
  public void string_literal_content() {
    UnaryOperator<String> stringLiteralContent = source ->
      CheckUtils.stringLiteralContent(PARSER.parse(source).getTokens().get(0).getOriginalValue());
    assertThat(stringLiteralContent.apply("\"abc\"")).isEqualTo("abc");
    assertThat(stringLiteralContent.apply("r''")).isEqualTo("");
    assertThat(stringLiteralContent.apply("fr\"abc abc\"")).isEqualTo("abc abc");
  }

  @Test(expected = IllegalStateException.class)
  public void invalid_string_literal_content() {
    CheckUtils.stringLiteralContent(PARSER.parse("2").getTokens().get(0).getOriginalValue());
  }

  @Test
  public void is_assignment_expression() {
    Predicate<String> firstStatement = source ->
      CheckUtils.isAssignmentExpression(PARSER.parse(source).getFirstDescendant(PythonGrammar.SIMPLE_STMT).getFirstChild());

    assertThat(firstStatement.test("a()")).isFalse();
    assertThat(firstStatement.test("a = 2")).isTrue();
    assertThat(firstStatement.test("a: int")).isFalse();
    assertThat(firstStatement.test("a: int = 2")).isTrue();
    assertThat(firstStatement.test("a.b = (1, 2)")).isTrue();
    assertThat(firstStatement.test("a.b: int = (1, 2)")).isTrue();
  }

  @Test
  public void statement_equivalence() {
    assertThat(CheckUtils.areEquivalent(parse("x = x + 1"), parse("x = x + 1"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x = x + 1"), parse("x = x + 1"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x = x + 1"), parse("x = x + 1"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x = x + 1"), parse("x = x + 2"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo()"), parse("foo()"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("foo()"), parse("foo"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo"), parse("foo()"))).isFalse();
  }

  @Test
  public void comparison_equivalence() {
    assertThat(CheckUtils.areEquivalent(parse("foo is None"), parse("foo is not None"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("x < 2"), parse("x > 2"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo is None"), parse("foo is  None"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x < 1"), parse("x < 1"))).isTrue();
  }

  @Test
  public void tree_equivalence() {
    assertThat(CheckUtils.areEquivalent(new PyArgListTreeImpl(null, Collections.emptyList()), new PyArgListTreeImpl(null, Collections.emptyList()))).isTrue();
  }

  @Test
  public void null_equivalence() {
    assertThat(CheckUtils.areEquivalent(null, null)).isTrue();
    assertThat(CheckUtils.areEquivalent(null, parse("class clazz(): \n pass"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("class clazz(): \n pass"), null)).isFalse();
  }

  @Test
  public void statement_list_equivalence() {
    assertThat(CheckUtils.areEquivalent(parse("foo()\nbar()"), parse("foo()\nbar()"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("foo()\n  "), parse("foo()\n"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("foo()\n"), parse("foo()\n  "))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("foo()"), parse("foo()\nfoo()"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo()\nfoo()"), parse("foo()"))).isFalse();
    assertThat(CheckUtils.areEquivalent(parse("foo()\nbar()"), parse("foo()\nbar"))).isFalse();
  }

  @Test
  public void lambda_equivalence() {
    assertThat(CheckUtils.areEquivalent(parse("x = lambda a : a + 10"), parse("x = lambda a : a + 10"))).isTrue();
    assertThat(CheckUtils.areEquivalent(parse("x = lambda a : a + 10"), parse("x = lambda a : a + 5"))).isFalse();
  }

  private Tree parse(String content) {
    Parser<Grammar> parser = PythonParser.create(new PythonConfiguration(StandardCharsets.UTF_8));
    AstNode astNode = parser.parse(content);
    PyFileInputTree parse = new PythonTreeMaker().fileInput(astNode);
    return parse;
  }
}
