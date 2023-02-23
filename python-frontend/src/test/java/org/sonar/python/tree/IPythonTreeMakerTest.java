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

import org.junit.Test;
import org.sonar.python.parser.RuleTest;

import static org.assertj.core.api.Assertions.assertThat;

public class IPythonTreeMakerTest extends RuleTest {

  private final IPythonTreeMaker treeMaker = new IPythonTreeMaker();

  @Test
  public void line_magic() {
    var parse = parseIPython("print(b)\n" +
      "a = %alias showPath pwd && ls -a\n", treeMaker::fileInput);
    assertThat(parse).isNotNull();

    parse = parseIPython("print(b)\n" +
      "%alias showPath pwd && ls -a\n", treeMaker::fileInput);
    assertThat(parse).isNotNull();

    parse = parseIPython("print(b)\n" +
      "%timeit print(a)\n", treeMaker::fileInput);
    assertThat(parse).isNotNull();

    parse = parseIPython("print(b)\n" +
      "a = %timeit foo(b)\n", treeMaker::fileInput);
    assertThat(parse).isNotNull();

    parse = parseIPython("print(b)\n" +
      "%autocall 1\n", treeMaker::fileInput);
    assertThat(parse).isNotNull();
  }
}
