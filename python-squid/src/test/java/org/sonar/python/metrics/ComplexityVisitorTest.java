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
package org.sonar.python.metrics;

import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.impl.Parser;
import java.io.File;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.parser.PythonParser;

import static org.fest.assertions.Assertions.assertThat;

public class ComplexityVisitorTest {

  private ComplexityVisitor visitor = new ComplexityVisitor();

  private Parser<Grammar> parser = PythonParser.create(new PythonConfiguration(StandardCharsets.UTF_8));

  @Test
  public void file() throws Exception {
    TestPythonVisitorRunner.scanFile(new File("src/test/resources/metrics/complexity.py"), visitor);
    assertThat(visitor.getComplexity()).isEqualTo(7);
  }

  @Test
  public void pass_keyword() throws Exception {
    assertThat(complexity("pass")).isEqualTo(0);
  }

  @Test
  public void if_keyword() throws Exception {
    assertThat(complexity("if x: pass")).isEqualTo(1);
  }

  @Test
  public void and_keyword() throws Exception {
    assertThat(complexity("x = a and b and c")).isEqualTo(2);
  }

  @Test
  public void or_keyword() throws Exception {
    assertThat(complexity("x = a or b or c")).isEqualTo(2);
  }

  @Test
  public void funcdef() throws Exception {
    assertThat(complexity("def f(): pass")).isEqualTo(1);
  }

  @Test
  public void while_statement() throws Exception {
    assertThat(complexity("while(x): pass")).isEqualTo(1);
  }

  @Test
  public void for_statement() throws Exception {
    assertThat(complexity("for i in list: pass")).isEqualTo(1);
  }

  private int complexity(String source) {
    return ComplexityVisitor.complexity(parser.parse(source));
  }

}
