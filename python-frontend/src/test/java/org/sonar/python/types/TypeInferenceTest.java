/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.types;

import java.util.List;
import org.junit.Test;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.SymbolTableBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.types.InferredTypes.BYTES;
import static org.sonar.python.types.InferredTypes.COMPLEX;
import static org.sonar.python.types.InferredTypes.DICT;
import static org.sonar.python.types.InferredTypes.FLOAT;
import static org.sonar.python.types.InferredTypes.GENERATOR;
import static org.sonar.python.types.InferredTypes.INT;
import static org.sonar.python.types.InferredTypes.LIST;
import static org.sonar.python.types.InferredTypes.SET;
import static org.sonar.python.types.InferredTypes.STR;
import static org.sonar.python.types.InferredTypes.TUPLE;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.runtimeType;

public class TypeInferenceTest {

  @Test
  public void unknown_expression_type() {
    assertThat(lastExpression("foo()").type()).isEqualTo(anyType());
  }

  @Test
  public void names() {
    assertThat(lastExpression("a = A()\na").type()).isEqualTo(anyType());
    assertThat(lastExpression("class A: pass\na = A()\na").type()).isEqualTo(runtimeType("mod1.A"));
  }

  @Test
  public void numeric_literals() {
    assertThat(lastExpression("42").type()).isEqualTo(INT);
    assertThat(lastExpression("42_3").type()).isEqualTo(INT);
    assertThat(lastExpression("0b101").type()).isEqualTo(INT);
    assertThat(lastExpression("0x1F").type()).isEqualTo(INT);
    assertThat(lastExpression("42.0").type()).isEqualTo(FLOAT);
    assertThat(lastExpression("1e100").type()).isEqualTo(FLOAT);
    assertThat(lastExpression("1E100").type()).isEqualTo(FLOAT);
    assertThat(lastExpression("42j").type()).isEqualTo(COMPLEX);
    assertThat(lastExpression("42.0j").type()).isEqualTo(COMPLEX);
  }

  @Test
  public void string_literals() {
    assertThat(lastExpression("'hello world'").type()).isEqualTo(STR);
    assertThat(lastExpression("f'hello world'").type()).isEqualTo(STR);
    assertThat(lastExpression("'hello' 'world'").type()).isEqualTo(STR);

    assertThat(lastExpression("b'hello'").type()).isEqualTo(BYTES);
    assertThat(lastExpression("rb'hello'").type()).isEqualTo(BYTES);

    // this throws a 'SyntaxError: cannot mix bytes and nonbytes literals'
    assertThat(lastExpression("b'hello' 'world'").type()).isEqualTo(STR);
  }

  @Test
  public void list_literals() {
    assertThat(lastExpression("[]").type()).isEqualTo(LIST);
    assertThat(lastExpression("[42]").type()).isEqualTo(LIST);

    assertThat(lastExpression("[x for x in range(0, 100)]").type()).isEqualTo(LIST);
  }

  @Test
  public void dict_literals() {
    assertThat(lastExpression("{}").type()).isEqualTo(DICT);
    assertThat(lastExpression("{'x' : 1, 'y' : 2}").type()).isEqualTo(DICT);

    assertThat(lastExpression("{ k:v for (k, v) in foo }").type()).isEqualTo(DICT);
  }

  @Test
  public void set_literals() {
    assertThat(lastExpression("{1, 2, 3}").type()).isEqualTo(SET);

    assertThat(lastExpression("{ v for v in foo }").type()).isEqualTo(SET);
  }

  @Test
  public void generator_literal() {
    assertThat(lastExpression("(v for v in foo)").type()).isEqualTo(GENERATOR);
  }

  @Test
  public void tuple_literal() {
    assertThat(lastExpression("()").type()).isEqualTo(TUPLE);
    assertThat(lastExpression("(1, 2)").type()).isEqualTo(TUPLE);
  }

  private Expression lastExpression(String code) {
    FileInput fileInput = PythonTestUtils.parse(new SymbolTableBuilder("", pythonFile("mod1.py")), code);
    List<Statement> statements = fileInput.statements().statements();
    Statement statement = statements.get(statements.size() - 1);
    assertThat(statement).isInstanceOf(ExpressionStatement.class);
    List<Expression> expressions = ((ExpressionStatement) statement).expressions();
    return expressions.get(expressions.size() - 1);
  }

}
