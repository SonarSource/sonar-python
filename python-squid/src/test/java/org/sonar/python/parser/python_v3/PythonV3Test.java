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
package org.sonar.python.parser.python_v3;

import org.junit.Test;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.parser.RuleTest;

import static org.sonar.sslr.tests.Assertions.assertThat;

public class PythonV3Test extends RuleTest {

  @Test
  public void ellipsis(){
    setRootRule(PythonGrammar.TEST);
    assertThat(p).matches("...");
    assertThat(p).matches("x[...]");
  }

  @Test
  public void matrix_multiplication(){
    setRootRule(PythonGrammar.STATEMENT);
    assertThat(p)
      .matches("a = b @ c")
      .matches("a = b @ c @ d")
      .matches("a @= b")
    ;
  }

  @Test
  public void function_declaration(){
    setRootRule(PythonGrammar.FUNCDEF);
    assertThat(p).matches("def fun()->'Returns some value': pass");
    assertThat(p).matches("def fun(count:'Number of smth'=1, value:'Value of smth', type:Type): pass");

    setRootRule(PythonGrammar.LAMBDEF);
    assertThat(p).matches("lambda x: x");
    assertThat(p).notMatches("lambda x:Type: x");
  }

  @Test
  public void asyncAndAwait() {
    setRootRule(PythonGrammar.FUNCDEF);
    assertThat(p).matches("async def fun(): pass");
    assertThat(p).matches("@someAnnotation\nasync def fun(): pass");

    setRootRule(PythonGrammar.ASYNC_STMT);
    assertThat(p).matches("async with x.bar() as foo: pass");
    assertThat(p).matches("async for var in x: pass");

    setRootRule(PythonGrammar.EXPRESSION_STMT);
    assertThat(p).matches("await async_func()");

    // before Python 3.5, 'async' and 'await' can be used as variable names or function names
    setRootRule(PythonGrammar.STATEMENT);
    assertThat(p).matches("async = 1");
    assertThat(p).matches("await = 1");
    setRootRule(PythonGrammar.FUNCDEF);
    assertThat(p).matches("def async(args): pass");
    assertThat(p).matches("def await(args): pass");
  }

  @Test
  public void yield_from(){
    setRootRule(PythonGrammar.YIELD_EXPR);
    assertThat(p).matches("yield");
    assertThat(p).matches("yield from foo");
    assertThat(p).matches("yield foo");
    assertThat(p).notMatches("yield from");
  }

  @Test
  public void function_star_parameters(){
    setRootRule(PythonGrammar.TYPEDARGSLIST);

    assertThat(p)
      .matches("*")
      .matches("*p")
      .matches("*p, p")
      .matches("*p, p = 1")
      .matches("*, p")
      .matches("*, p=1")
      .matches("*p, p, p")
      .matches("*p, p, **p")
      .matches("*p, **p")
      .matches("*p, p=1, **p")
      .matches("*p, p, p=1, **p")
      .matches("*p, p=1, p, **p")
      .matches("**p")
      .matches("p")
      .matches("p,")
      .matches("p=1")
      .matches("p=1, p")
      .matches("p, p=1")
      .matches("p=1, p=1")
      .matches("p, **p")
      .matches("p, p, **p")
      .matches("p=1, p=1, **p")
      .matches("p, *")
      .matches("p, *, p=1, p")
      .matches("p, *p")
      .matches("p, *p, p")
      .matches("p, *p, p, **p")
      .matches("p, *, p, p, **p")
      .matches("p, *p, **p")

      .notMatches("p *p")
    ;
  }

  @Test
  public void unpacking_operations() throws Exception {
    setRootRule(PythonGrammar.ARGLIST);

    assertThat(p)
      .matches("*[1]")
      .matches("*[1], *[2], 3")
      .matches("**{'x': 1}, y=2, **{'z': 3}")
    ;


    setRootRule(PythonGrammar.TESTLIST_STAR_EXPR);

    assertThat(p)
      // tuple
      .matches("*a, b")
      .matches("(a, *b)")

      // list
      .matches("[*a, b]")

      // set
      .matches("{*a, b}")

      // dictionary
      .matches("{'x':b, **a}")
    ;
  }
}
