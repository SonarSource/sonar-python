/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.parser.simple_statements;

import com.google.common.base.Charsets;
import com.sonar.sslr.impl.Parser;
import org.junit.Before;
import org.junit.Test;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.PythonParser;
import org.sonar.sslr.tests.Assertions;

import static org.sonar.sslr.tests.Assertions.assertThat;

public class SimpleStatementTest {

  Parser<PythonGrammar> p = PythonParser.create(new PythonConfiguration(Charsets.UTF_8));
  PythonGrammar g = p.getGrammar();

  @Before
  public void init() {
    p.setRootRule(g.simple_stmt);
  }

  @Test
  public void ok() {
    assertThat(p).matches("print_stmt");
    assertThat(p).matches("exec_stmt");
    assertThat(p).matches("expression_stmt");
    assertThat(p).matches("assert_stmt");
    assertThat(p).matches("pass_stmt");
    assertThat(p).matches("del_stmt");
    assertThat(p).matches("return_stmt");
    assertThat(p).matches("yield_stmt");
    assertThat(p).matches("raise_stmt");
    assertThat(p).matches("break_stmt");
    assertThat(p).matches("continue_stmt");
    assertThat(p).matches("import_stmt");
    assertThat(p).matches("global_stmt");
    assertThat(p).matches("nonlocal_stmt");
  }

  @Test
  public void realLife() {
    assertThat(p).matches("print 'Hello world'");
    assertThat(p).matches("exec 'print 1'");
    assertThat(p).matches("i += 1");
    assertThat(p).matches("print('something', file=out_file)");
  }

}
