/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
import java.util.Set;
import java.util.function.Function;
import org.junit.Test;
import org.sonar.python.PythonCheck;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.parser.PythonParser;

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

  private static class EqualsNodeCheck extends PythonCheck {
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

  private static class IsMethodDefinitionCheck extends PythonCheck {
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
    PythonCheckVerifier.verify("src/test/resources/checks/utils/classHasInheritance.py", new classHasInheritanceCheck());
  }

  private static class classHasInheritanceCheck extends PythonCheck {
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
  public void string_interpolation() throws Exception {
    Function<String, Boolean> isStringInterpolation = (source) -> CheckUtils.isStringInterpolation(PARSER.parse(source).getTokens().get(0));
    assertThat(isStringInterpolation.apply("\"abc\"")).isFalse();
    assertThat(isStringInterpolation.apply("r'abc'")).isFalse();
    assertThat(isStringInterpolation.apply("f'abc'")).isTrue();
    assertThat(isStringInterpolation.apply("Rf'abc'")).isTrue();
    assertThat(isStringInterpolation.apply("rF'abc'")).isTrue();
    assertThat(isStringInterpolation.apply("fr\"\"")).isTrue();
  }

  @Test
  public void string_literal_content() throws Exception {
    Function<String, String> stringLiteralContent = (source) ->
      CheckUtils.stringLiteralContent(PARSER.parse(source).getTokens().get(0).getOriginalValue());
    assertThat(stringLiteralContent.apply("\"abc\"")).isEqualTo("abc");
    assertThat(stringLiteralContent.apply("r''")).isEqualTo("");
    assertThat(stringLiteralContent.apply("fr\"abc abc\"")).isEqualTo("abc abc");
  }

  @Test(expected = IllegalStateException.class)
  public void invalid_string_literal_content() throws Exception {
    CheckUtils.stringLiteralContent(PARSER.parse("2").getTokens().get(0).getOriginalValue());
  }

  @Test
  public void is_assignment_expression() throws Exception {
    Function<String, Boolean> firstStatement = (source) ->
      CheckUtils.isAssignmentExpression(PARSER.parse(source).getFirstDescendant(PythonGrammar.SIMPLE_STMT).getFirstChild());

    assertThat(firstStatement.apply("a()")).isFalse();
    assertThat(firstStatement.apply("a = 2")).isTrue();
    assertThat(firstStatement.apply("a: int")).isFalse();
    assertThat(firstStatement.apply("a: int = 2")).isTrue();
    assertThat(firstStatement.apply("a.b = (1, 2)")).isTrue();
    assertThat(firstStatement.apply("a.b: int = (1, 2)")).isTrue();
  }
}
