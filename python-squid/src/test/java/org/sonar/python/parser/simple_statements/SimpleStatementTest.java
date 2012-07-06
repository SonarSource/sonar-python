/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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

import com.sonar.sslr.impl.Parser;
import org.junit.Before;
import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.PythonParser;

import static com.sonar.sslr.test.parser.ParserMatchers.parse;
import static org.junit.Assert.assertThat;

public class SimpleStatementTest {

  Parser<PythonGrammar> p = PythonParser.create();
  PythonGrammar g = p.getGrammar();

  @Before
  public void init() {
    p.setRootRule(g.simple_stmt);
  }

  @Test
  public void ok() {
    g.print_stmt.mock();
    g.exec_stmt.mock();
    g.expression_stmt.mock();
    g.assert_stmt.mock();
    g.pass_stmt.mock();
    g.del_stmt.mock();
    g.return_stmt.mock();
    g.yield_stmt.mock();
    g.raise_stmt.mock();
    g.break_stmt.mock();
    g.continue_stmt.mock();
    g.import_stmt.mock();
    g.global_stmt.mock();
    g.nonlocal_stmt.mock();

    assertThat(p, parse("print_stmt"));
    assertThat(p, parse("exec_stmt"));
    assertThat(p, parse("expression_stmt"));
    assertThat(p, parse("assert_stmt"));
    assertThat(p, parse("pass_stmt"));
    assertThat(p, parse("del_stmt"));
    assertThat(p, parse("return_stmt"));
    assertThat(p, parse("yield_stmt"));
    assertThat(p, parse("raise_stmt"));
    assertThat(p, parse("break_stmt"));
    assertThat(p, parse("continue_stmt"));
    assertThat(p, parse("import_stmt"));
    assertThat(p, parse("global_stmt"));
    assertThat(p, parse("nonlocal_stmt"));
  }

  @Test
  public void realLife() {
    assertThat(p, parse("print 'Hello world'"));
    assertThat(p, parse("exec 'print 1'"));
    assertThat(p, parse("i += 1"));
    assertThat(p, parse("print('something', file=out_file)"));
  }

}
