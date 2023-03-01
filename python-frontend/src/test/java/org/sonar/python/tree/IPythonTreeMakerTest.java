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
package org.sonar.python.tree;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.Test;
import org.sonar.plugins.python.api.tree.LineMagic;
import org.sonar.plugins.python.api.tree.LineMagicStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.parser.RuleTest;

import static org.assertj.core.api.Assertions.assertThat;

public class IPythonTreeMakerTest extends RuleTest {

  private final IPythonTreeMaker treeMaker = new IPythonTreeMaker();

  @Test
  public void line_magic() {
    var parse = parseIPython("print(b)\n" +
      "a = %alias showPath pwd && ls -a\n", treeMaker::fileInput);
    assertThat(parse).isNotNull();
    var lineMagic = findFirstChildOf(parse, Tree.Kind.LINE_MAGIC);
    assertThat(lineMagic)
      .isNotNull()
      .isInstanceOf(LineMagic.class);

    parse = parseIPython("print(b)\n" +
      "a = %timeit foo(b)\n", treeMaker::fileInput);
    assertThat(parse).isNotNull();
    lineMagic = findFirstChildOf(parse, Tree.Kind.LINE_MAGIC);
    assertThat(lineMagic)
      .isNotNull()
      .isInstanceOf(LineMagic.class);

    parse = parseIPython("print(b)\n" +
      "a = %timeit foo(b) % 3\n" +
      "print(a)", treeMaker::fileInput);
    assertThat(parse).isNotNull();
    lineMagic = findFirstChildOf(parse, Tree.Kind.LINE_MAGIC);
    assertThat(lineMagic)
      .isNotNull()
      .isInstanceOf(LineMagic.class);
  }

  @Test
  public void line_magic_statement() {
    var parse = parseIPython("print(b)\n" +
      "%alias showPath pwd && ls -a\n", treeMaker::fileInput);
    assertThat(parse).isNotNull();
    var lineMagicStatement = findFirstChildOf(parse, Tree.Kind.LINE_MAGIC_STATEMENT);
    assertThat(lineMagicStatement)
      .isNotNull()
      .isInstanceOf(LineMagicStatement.class);

    assertThat(lineMagicStatement.children()).hasSize(1);
    var lineMagic = findFirstChildOf(lineMagicStatement, Tree.Kind.LINE_MAGIC);
    assertThat(lineMagic).isNotNull();

    parse = parseIPython("print(b)\n" +
      "%timeit a = foo(b) % 3\n" +
      "a %= 2\n" +
      "print(a)", treeMaker::fileInput);
    assertThat(parse).isNotNull();
    lineMagicStatement = findFirstChildOf(parse, Tree.Kind.LINE_MAGIC_STATEMENT);
    assertThat(lineMagicStatement)
      .isNotNull()
      .isInstanceOf(LineMagicStatement.class);

    assertThat(lineMagicStatement.children()).hasSize(1);
    lineMagic = findFirstChildOf(lineMagicStatement, Tree.Kind.LINE_MAGIC);
    assertThat(lineMagic).isNotNull();

    parse = parseIPython("print(b)\n" +
      "%timeit a = foo(b); b = 2\n" +
      "a += b\n" +
      "print(a)", treeMaker::fileInput);
    assertThat(parse).isNotNull();
    lineMagicStatement = findFirstChildOf(parse, Tree.Kind.LINE_MAGIC_STATEMENT);
    assertThat(lineMagicStatement)
      .isNotNull()
      .isInstanceOf(LineMagicStatement.class);

    assertThat(lineMagicStatement.children()).hasSize(1);
    lineMagic = findFirstChildOf(lineMagicStatement, Tree.Kind.LINE_MAGIC);
    assertThat(lineMagic).isNotNull();

    parse = parseIPython("print(b)\n" +
      "%autocall 1\n", treeMaker::fileInput);
    assertThat(parse).isNotNull();
  }

  @Test
  public void assignment_rhs_test() {
    var parse = parseIPython("print(b)\n" +
      "a = yield foo(b)\n" +
      "c = bar(a) + b", treeMaker::fileInput);
    assertThat(parse).isNotNull();

    var assignments = findChildrenOf(parse, Tree.Kind.ASSIGNMENT_STMT);
    assertThat(assignments).hasSize(2);
    var yieldExpression = findFirstChildOf(assignments.get(0), Tree.Kind.YIELD_EXPR);
    assertThat(yieldExpression).isNotNull();
    var binaryExpression = findFirstChildOf(assignments.get(1), Tree.Kind.PLUS);
    assertThat(binaryExpression).isNotNull();
  }

  private <T extends Tree> List<T> findChildrenOf(Tree parent, Tree.Kind kind) {
    return (List<T>) parent.children()
      .stream()
      .flatMap(c -> {
        if (c.is(kind)) {
          return Stream.of(c);
        } else {
          return findChildrenOf(c, kind).stream();
        }
      })
      .collect(Collectors.toList());
  }

  private <T extends Tree> T findFirstChildOf(Tree parent, Tree.Kind kind) {
    return (T) parent.children()
      .stream()
      .map(c -> {
        if (c.is(kind)) {
          return c;
        } else {
          return findFirstChildOf(c, kind);
        }
      })
      .filter(Objects::nonNull)
      .findFirst()
      .orElse(null);
  }
}
