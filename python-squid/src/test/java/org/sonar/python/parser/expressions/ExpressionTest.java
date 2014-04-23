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
package org.sonar.python.parser.expressions;

import com.google.common.base.Charsets;
import com.sonar.sslr.impl.Parser;
import org.junit.Before;
import org.junit.Test;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.PythonParser;

import static com.sonar.sslr.test.parser.ParserMatchers.parse;
import static org.junit.Assert.assertThat;

public class ExpressionTest {

  Parser<PythonGrammar> p = PythonParser.create(new PythonConfiguration(Charsets.UTF_8));
  PythonGrammar g = p.getGrammar();

  @Before
  public void init() {
    p.setRootRule(g.test);
  }

  @Test
  public void realLife() {
    assertThat(p, parse("1 + 2 * 3"));
    assertThat(p, parse("(1 + 1) * 2"));

    assertThat(p, parse("True"));
    assertThat(p, parse("False"));
    assertThat(p, parse("None"));

    assertThat(p, parse("list[1]"));
    assertThat(p, parse("list[1:3]"));
    assertThat(p, parse("list[:]"));

    assertThat("list", p, parse("[1, 2]"));
    assertThat("list with trailing comma", p, parse("[1, 2,]"));

    assertThat("dictionary", p, parse("{'foo': 1, 'bar': 2, 'baz': 3,}"));
    assertThat("dictionary with trailing comma", p, parse("{'foo': 1, 'bar': 2, 'baz': 3,}"));

    assertThat("trailing comma", p, parse("print(something,)"));

    assertThat(p, parse("func(value, parameter = value)"));

    assertThat(p, parse("lambda x: x**2"));
  }

}
