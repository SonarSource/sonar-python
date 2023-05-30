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
package org.sonar.python.types;

import java.util.List;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.getLastDescendant;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.lastExpressionInFunction;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.types.InferredTypes.BOOL;
import static org.sonar.python.types.InferredTypes.COMPLEX;
import static org.sonar.python.types.InferredTypes.DECL_INT;
import static org.sonar.python.types.InferredTypes.DECL_STR;
import static org.sonar.python.types.InferredTypes.DICT;
import static org.sonar.python.types.InferredTypes.FLOAT;
import static org.sonar.python.types.InferredTypes.INT;
import static org.sonar.python.types.InferredTypes.LIST;
import static org.sonar.python.types.InferredTypes.NONE;
import static org.sonar.python.types.InferredTypes.SET;
import static org.sonar.python.types.InferredTypes.STR;
import static org.sonar.python.types.InferredTypes.TUPLE;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.or;
import static org.sonar.python.types.InferredTypes.runtimeType;
import static org.sonar.python.types.InferredTypes.typeName;

public class TypeInferenceTest {

  @Test
  public void unknown_expression_type() {
    assertThat(lastExpression("a.b").type()).isEqualTo(anyType());
    assertThat(lastExpression("a[0]").type()).isEqualTo(anyType());
  }

  @Test
  public void unpacking_assignment() {
    assertThat(lastExpressionInFunction(
      "x, = 42,",
      "x"
    ).type()).isEqualTo(anyType());
  }

  @Test
  public void call_expression() {
    assertThat(lastExpression(
      "f()").type()).isEqualTo(anyType());
    assertThat(lastExpression(
      "def f(): pass",
      "f()").type()).isEqualTo(anyType());

    CallExpression expression = (CallExpression) lastExpression(
      "class A: pass",
      "A()");
    assertThat(expression.calleeSymbol().fullyQualifiedName()).isEqualTo("mod1.A");
    assertThat(expression.type()).isEqualTo(runtimeType((ClassSymbol) expression.calleeSymbol()));
  }

  @Test
  public void variable_outside_function() {
    assertThat(lastExpression("a = 42; a").type()).isEqualTo(anyType());
  }

  @Test
  public void parameter() {
    assertThat(lastExpression(
      "def f(p):",
      "  p = 42",
      "  p").type()).isEqualTo(anyType());

    assertThat(lastExpression(
      "def f(p):",
      "  a = p",
      "  if cond:",
      "    a = 42",
      "  a").type()).isEqualTo(anyType());
  }

  @Test
  public void parameter_with_annotation() {
    assertThat(lastExpression("def f(p: int): p").type()).isEqualTo(DECL_INT);
    assertThat(lastExpression("def f(p: str): p").type()).isEqualTo(DECL_STR);
    InferredType typeA = lastExpression("class A: ...\ndef f(p: A): p").type();
    assertThat(typeA).isInstanceOf(DeclaredType.class);
    assertThat(typeName(typeA)).isEqualTo("A");
    assertThat(lastExpression("def f(p: unknown): p").type()).isEqualTo(anyType());
    assertThat(lastExpression("def f(p1: int, *, p2: str): p2").type()).isEqualTo(DECL_STR);

    assertThat(lastExpression(
      "def f(p: int):",
      "  p = 'str'",
      "  p").type()).isEqualTo(STR);

    assertThat(lastExpression(
      "def f(p: int):",
      "  try: ...",
      "  except: ...",
      "  p").type()).isEqualTo(anyType());

    assertThat(lastExpression("def f(*p: int): p").type()).isEqualTo(TUPLE);
    assertThat(lastExpression("def f(**p: int): p").type()).isEqualTo(DICT);
  }

  @Test
  public void local_variable() {
    assertThat(lastExpressionInFunction(
      "a = 42",
      "a").type()).isEqualTo(INT);
  }

  @Test
  public void global_variable() {
    assertThat(lastExpressionInFunction(
      "global a",
      "a = 42",
      "a").type()).isEqualTo(anyType());
  }

  @Test
  public void simple_propagation_between_variables() {
    assertThat(lastExpressionInFunction(
      "a = ''",
      "b = a",
      "c = b",
      "c").type()).isEqualTo(STR);
  }

  @Test
  public void variable_read_appearing_before_initialization() {
    assertThat(lastExpressionInFunction(
      "for i in range(3):",
      "  if i > 0: a = b",
      "  else:     b = 1",
      "a").type()).isEqualTo(INT);
  }

  @Test
  public void cycle_between_variables_with_initialization() {
    assertThat(lastExpressionInFunction(
      "for i in range(3):",
      "  if i > 1:  b = a",
      "  elif i==1: a = b",
      "  else:      b = 1",
      "a").type()).isEqualTo(INT);
  }

  @Test
  public void unresolvable_cycle_between_variables() {
    assertThat(lastExpressionInFunction(
      "if cond: a = b",
      "else:    b = a",
      "c = 1",
      "a").type()).isEqualTo(anyType());
  }

  @Test
  public void unsupported_assignment() {
    assertThat(lastExpressionInFunction(
      "(a, b) = foo()",
      "a").type()).isEqualTo(anyType());

    assertThat(lastExpressionInFunction(
      "(a, b) = foo()",
      "a = ''",
      "a").type()).isEqualTo(anyType());

    assertThat(lastExpressionInFunction(
      "(a, b) = foo()",
      "a = ''",
      "c = a",
      "c").type()).isEqualTo(anyType());

    assertThat(lastExpressionInFunction(
      "c = 42",
      "if cond: a, b = foo()",
      "else:    a = c",
      "d = a",
      "d").type()).isEqualTo(anyType());
  }

  @Test
  public void propagate_return_type_to_variable() {
    assertThat(lastExpressionInFunction(
      "a = 'abc'.capitalize()",
      "a").type()).isEqualTo(STR);

    assertThat(lastExpressionInFunction(
      "for i in range(3):",
      "  if i > 0: b = a.capitalize()",
      "  else:     a = 'abc'",
      "b").type()).isEqualTo(STR);

    assertThat(lastExpressionInFunction(
      "for i in range(3):",
      "  if i > 0: b = a.capitalize().upper()",
      "  else:     a = 'abc'",
      "b").type()).isEqualTo(STR);

    assertThat(lastExpressionInFunction(
      "if cond:  a = 'abc'",
      "else:     a = x.foo()",
      "b = a.capitalize()",
      "b").type()).isEqualTo(anyType());

    assertThat(lastExpressionInFunction(
      "global a",
      "for i in range(3):",
      "  if i > 0: b = a.capitalize()",
      "  else:     a = 'abc'",
      "b").type()).isEqualTo(anyType());
  }

  @Test
  public void multiple_types() {
    assertThat(lastExpressionInFunction(
      "if cond: a = ''",
      "else:    a = 42",
      "a").type()).isEqualTo(or(STR, INT));
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

    assertThat(lastExpression("b'hello'").type()).isEqualTo(anyType());
    assertThat(lastExpression("rb'hello'").type()).isEqualTo(anyType());

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
    assertThat(lastExpression("(v for v in foo)").type()).isEqualTo(anyType());
  }

  @Test
  public void tuple_literal() {
    assertThat(lastExpression("()").type()).isEqualTo(TUPLE);
    assertThat(lastExpression("(1, 2)").type()).isEqualTo(TUPLE);
  }

  @Test
  public void none_type() {
    assertThat(lastExpression("None").type()).isEqualTo(NONE);
  }

  @Test
  public void true_false_literal() {
    assertThat(lastExpression("True").type()).isEqualTo(BOOL);
    assertThat(lastExpression("False").type()).isEqualTo(BOOL);
  }

  @Test
  public void builtin_function_types() {
    assertThat(lastExpression("all([1, 2, 3])").type()).isEqualTo(BOOL);
    assertThat(lastExpression("round(42)").type()).isEqualTo(AnyType.ANY);
    ClassSymbol classSymbolRange = ((ClassSymbol) TypeShed.builtinSymbols().get("range"));
    assertThat(lastExpression("range(42)").type()).isEqualTo(InferredTypes.runtimeType(classSymbolRange));
    assertThat(lastExpression("getattr(42)").type()).isEqualTo(AnyType.ANY);
  }

  @Test
  public void builtin_method_types() {
    assertThat(lastExpression("'abc'.capitalize()").type()).isEqualTo(STR);
    assertThat(lastExpression("list().copy()").type()).isEqualTo(LIST);
  }

  @Test
  public void conditional_expressions() {
    assertThat(lastExpression("42 if '' else 43").type()).isEqualTo(INT);
    assertThat(lastExpression("42 if cond else ''").type()).isEqualTo(or(INT, STR));
    assertThat(lastExpression("42 if '' else xxx").type()).isEqualTo(anyType());
    assertThat(lastExpression("42 if cond1 else True if cond2 else ''").type()).isEqualTo(or(or(INT, STR), BOOL));

    assertThat(lastExpressionInFunction(
      "for i in range(3):",
      "  if   i > 1: c = b",
      "  elif i > 0: b = 42 if cond else a",
      "  else:       a = ''",
      "c").type()).isEqualTo(or(INT, STR));

    assertThat(lastExpressionInFunction(
      "c = 42 if '' else c",
      "c").type()).isEqualTo(anyType());
  }

  @Test
  public void flow_sensitive_type_inference() {
    assertThat(lastExpressionInFunction(
      "x = 42",
      "x = '42'",
      "x"
    ).type()).isEqualTo(STR);


    FileInput fileInput = parse(
      "def f(p):",
      "  if p:",
      "    x = 42",
      "    type(x)",
      "  else:",
      "    x = 'foo'",
      "    type(x)",
      "  type(x)"
    );
    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument firstX = (RegularArgument) calls.get(0).arguments().get(0);
    RegularArgument secondX = (RegularArgument) calls.get(1).arguments().get(0);
    RegularArgument thirdX = (RegularArgument) calls.get(2).arguments().get(0);
    assertThat(firstX.expression().type()).isEqualTo(INT);
    assertThat(secondX.expression().type()).isEqualTo(STR);
    assertThat(thirdX.expression().type()).isEqualTo(or(INT, STR));
  }

  @Test
  public void flow_insensitive_when_try_except() {
    FileInput fileInput = parse(
      "def f(p):",
      "  try:",
      "    if p:",
      "      x = 42",
      "      type(x)",
      "    else:",
      "      x = 'foo'",
      "      type(x)",
      "  except:",
      "    type(x)"
    );
    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument firstX = (RegularArgument) calls.get(0).arguments().get(0);
    RegularArgument secondX = (RegularArgument) calls.get(1).arguments().get(0);
    RegularArgument thirdX = (RegularArgument) calls.get(2).arguments().get(0);
    assertThat(firstX.expression().type()).isEqualTo(or(INT, STR));
    assertThat(secondX.expression().type()).isEqualTo(or(INT, STR));
    assertThat(thirdX.expression().type()).isEqualTo(or(INT, STR));
  }

  @Test
  public void nested_try_except() {
    FileInput fileInput = parse(
      "def func(cond):",
      "  def f(p):",
      "    try:",
      "      if p:",
      "        x = 42",
      "        type(x)",
      "      else:",
      "        x = 'foo'",
      "        type(x)",
      "    except:",
      "      type(x)",
      "  def g(p):",
      "    if p:",
      "      y = 42",
      "      type(y)",
      "    else:",
      "      y = \"hello\"",
      "      type(y)",
      "    type(y)",
      "  if cond:",
      "    z = 42",
      "    type(z)",
      "  else:",
      "    z = \"hello\"",
      "    type(z)",
      "  type(z)"
    );
    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument firstX = (RegularArgument) calls.get(0).arguments().get(0);
    RegularArgument secondX = (RegularArgument) calls.get(1).arguments().get(0);
    RegularArgument thirdX = (RegularArgument) calls.get(2).arguments().get(0);
    assertThat(firstX.expression().type()).isEqualTo(or(INT, STR));
    assertThat(secondX.expression().type()).isEqualTo(or(INT, STR));
    assertThat(thirdX.expression().type()).isEqualTo(or(INT, STR));

    RegularArgument firstY = (RegularArgument) calls.get(3).arguments().get(0);
    RegularArgument secondY = (RegularArgument) calls.get(4).arguments().get(0);
    RegularArgument thirdY = (RegularArgument) calls.get(5).arguments().get(0);
    assertThat(firstY.expression().type()).isEqualTo(INT);
    assertThat(secondY.expression().type()).isEqualTo(STR);
    assertThat(thirdY.expression().type()).isEqualTo(or(INT, STR));

    RegularArgument firstZ = (RegularArgument) calls.get(6).arguments().get(0);
    RegularArgument secondZ = (RegularArgument) calls.get(7).arguments().get(0);
    RegularArgument thirdZ = (RegularArgument) calls.get(8).arguments().get(0);
    assertThat(firstZ.expression().type()).isEqualTo(INT);
    assertThat(secondZ.expression().type()).isEqualTo(STR);
    assertThat(thirdZ.expression().type()).isEqualTo(or(INT, STR));
  }

  @Test
  public void nested_try_except_2() {
    FileInput fileInput = parse(
      "def func(cond):",
      "  try:",
      "    if p:",
      "      x = 42",
      "      type(x)",
      "    else:",
      "      x = 'foo'",
      "      type(x)",
      "  except:",
      "    type(x)",
      "  def g(p):",
      "    if p:",
      "      y = 42",
      "      type(y)",
      "    else:",
      "      y = \"hello\"",
      "      type(y)",
      "    type(y)",
      "  if cond:",
      "    z = 42",
      "    type(z)",
      "  else:",
      "    z = \"hello\"",
      "    type(z)",
      "  type(z)"
    );
    List<CallExpression> calls = PythonTestUtils.getAllDescendant(fileInput, tree -> tree.is(Tree.Kind.CALL_EXPR));
    RegularArgument firstX = (RegularArgument) calls.get(0).arguments().get(0);
    RegularArgument secondX = (RegularArgument) calls.get(1).arguments().get(0);
    RegularArgument thirdX = (RegularArgument) calls.get(2).arguments().get(0);
    assertThat(firstX.expression().type()).isEqualTo(or(INT, STR));
    assertThat(secondX.expression().type()).isEqualTo(or(INT, STR));
    assertThat(thirdX.expression().type()).isEqualTo(or(INT, STR));

    RegularArgument firstY = (RegularArgument) calls.get(3).arguments().get(0);
    RegularArgument secondY = (RegularArgument) calls.get(4).arguments().get(0);
    RegularArgument thirdY = (RegularArgument) calls.get(5).arguments().get(0);
    assertThat(firstY.expression().type()).isEqualTo(INT);
    assertThat(secondY.expression().type()).isEqualTo(STR);
    assertThat(thirdY.expression().type()).isEqualTo(or(INT, STR));

    RegularArgument firstZ = (RegularArgument) calls.get(6).arguments().get(0);
    RegularArgument secondZ = (RegularArgument) calls.get(7).arguments().get(0);
    RegularArgument thirdZ = (RegularArgument) calls.get(8).arguments().get(0);
    assertThat(firstZ.expression().type()).isEqualTo(or(INT, STR));
    assertThat(secondZ.expression().type()).isEqualTo(or(INT, STR));
    assertThat(thirdZ.expression().type()).isEqualTo(or(INT, STR));
  }

  @Test
  public void execution_order_assignment_statement() {
    FileInput fileInput = parse(
      "def foo():",
      "  x = 42",
      "  x = str(x)"
    );
    AssignmentStatement assignment = getLastDescendant(fileInput, tree -> tree.is(Tree.Kind.ASSIGNMENT_STMT));
    CallExpression call = (CallExpression) assignment.assignedValue();
    RegularArgument xRhs = (RegularArgument) call.arguments().get(0);
    assertThat(xRhs.expression().type()).isEqualTo(INT);

    Expression xLhs = assignment.lhsExpressions().get(0).expressions().get(0);
    assertThat(xLhs.type()).isEqualTo(STR);
  }

  @Test
  public void isinstance_flow_sensitive() {
    assertThat(lastExpression(
      "def f(x: int):",
      "  if isinstance(x, Foo):",
      "    ...",
      "  x"
      ).type()).isEqualTo(anyType());

    assertThat(lastExpression(
      "def f(x: int):",
      "  if not isinstance(x, Foo):",
      "    ...",
      "  x"
    ).type()).isEqualTo(anyType());

    FileInput fileInput = parse(
      "def f(x: int):",
      "  if isinstance(x, Foo):",
      "    x"
    );
    ExpressionStatement expressionStatement = getLastDescendant(fileInput, tree -> tree.is(Tree.Kind.EXPRESSION_STMT));
    Expression x = expressionStatement.expressions().get(0);
    assertThat(x.type()).isEqualTo(anyType());

    assertThat(lastExpression(
      "def f(x: int):",
      "  if isinstance(x):",
      "    ...",
      "  x"
    ).type()).isEqualTo(DECL_INT);

    assertThat(lastExpression(
      "def f(x: int):",
      "  if isinstance(foo(), Foo):",
      "    ...",
      "  x"
    ).type()).isEqualTo(DECL_INT);

    assertThat(lastExpression(
      "def f(x: int):",
      "  if unknown(x, Foo):",
      "    ...",
      "  x"
    ).type()).isEqualTo(DECL_INT);

    assertThat(lastExpression(
      "def f(x: int):",
      "  vars = [x]",
      "  if isinstance(*vars, Foo):",
      "    ...",
      "  x"
    ).type()).isEqualTo(DECL_INT);

    assertThat(lastExpressionInFunction(
      "x = 42",
      "if isinstance(x, Foo): ...",
      "x"
    ).type()).isEqualTo(INT);
  }

  @Test
  public void isinstance_flow_insensitive() {
    assertThat(lastExpression(
      "def f(x: int):",
      "  try:",
      "    if isinstance(x, Foo): ...",
      "  except: ...",
      "  x"
    ).type()).isEqualTo(anyType());
  }

  @Test
  public void typeshed_attributes() {
    assertThat(lastExpression(
      "def f():",
      "  e = OSError()",
      "  e.errno.bit_length()"
    ).type()).isEqualTo(INT);
  }

  @Test
  public void user_defined_attributes() {
    assertThat(lastExpression(
      "class Foo:",
      "  attr: int",
      "def f():",
      "  e = Foo()",
      "  e.attr.bit_length()"
    ).type()).isEqualTo(anyType());
  }

}
