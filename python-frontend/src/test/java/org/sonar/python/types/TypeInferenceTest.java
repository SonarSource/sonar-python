/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.types;

import java.util.List;
import org.assertj.core.groups.Tuple;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.semantic.SymbolTableBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.getLastDescendant;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.lastExpressionInFunction;
import static org.sonar.python.PythonTestUtils.lastStatement;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.PythonTestUtils.pythonFile;
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
import static org.sonar.python.types.InferredTypes.TYPE;
import static org.sonar.python.types.InferredTypes.anyType;
import static org.sonar.python.types.InferredTypes.or;
import static org.sonar.python.types.InferredTypes.runtimeType;
import static org.sonar.python.types.InferredTypes.typeName;

class TypeInferenceTest {

  @Test
  void unknown_expression_type() {
    assertThat(lastExpression("a.b").type()).isEqualTo(anyType());
    assertThat(lastExpression("a[0]").type()).isEqualTo(anyType());
  }

  @Test
  void unpacking_assignment() {
    assertThat(lastExpressionInFunction(
      "x, = 42,",
      "x"
    ).type()).isEqualTo(anyType());
  }

  @Test
  void genericsInV1() {
    Expression expression = lastExpression(
      """
        x = list[str]()
        x
        """
    );
    InferredType type = expression.type();
    assertThat(type.canOnlyBe("list")).isTrue();
  }

  @Test
  void call_expression() {
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
  void unknown_class_type() {
    var expression = (CallExpression) lastExpression(
      "def oauth2_session(a):",
        "  from requests_oauthlib.oauth2_session import OAuth2Session",
        "  oauth = OAuth2Session()",
        "  oauth.fetch_token()");
    assertThat(expression.calleeSymbol().fullyQualifiedName()).isEqualTo("requests_oauthlib.oauth2_session.OAuth2Session.fetch_token");

    expression = (CallExpression) lastExpression(
      "def oauth2_session(a):",
        "  from requests_oauthlib.oauth2_session import OAuth2Session",
        "  if a:",
        "    OAuth2Session = \"a\"",
        "  oauth = OAuth2Session()",
        "  oauth.fetch_token()");
    assertThat(expression.calleeSymbol().fullyQualifiedName()).isNull();
  }

  @Test
  void variable_outside_function() {
    assertThat(lastExpression("a = 42; a").type()).isEqualTo(INT);
  }
  @Test
  void variable_outside_function_2() {
    assertThat(lastExpression(
      "a = 42",
      "def foo():",
      "  a").type()).isEqualTo(anyType());
  }

  @Test
  void variable_outside_function_3() {
    assertThat(lastExpression(
      "def foo():",
      "  a = 42",
      "a").type()).isEqualTo(anyType());
  }

  @Test
  void variable_outside_function_4() {
    assertThat(lastExpression(
      "a = 42",
      "def foo():",
      "  a = 'hello'",
      "a").type()).isEqualTo(INT);
  }

  @Test
  void parameter() {
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
  void parameter_with_annotation() {
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
  void assignement_with_annotation() {
    assertThat(lastExpression(
      "a: str = \"foo\"",
      "a").type()).isEqualTo(STR);

    // We do not trust the type annotation
    assertThat(lastExpression(
      "a: int = {'foo', 'bar'}",
      "a").type()).isEqualTo(SET);
  }

  @Test
  void annotation_with_unknown_type() {
    assertThat(lastExpression(
      "def foo(unknown):",
      "  a: str = unknown()",
      "  a").type()).isEqualTo(anyType());
  }

  @Test
  void annotation_with_reassignment() {
    assertThat(lastExpression(
      "a = \"foo\"",
      "b: int = a",
      "b").type()).isEqualTo(STR);
  }

  @Test
  void annotation_without_assignement() {
    // We do not trust the type annotation
    assertThat(lastExpression(
      "a: str",
      "a").type()).isEqualTo(anyType());
  }

  @Test
  void local_variable() {
    assertThat(lastExpressionInFunction(
      "a = 42",
      "a").type()).isEqualTo(INT);
  }

  @Test
  void global_variable() {
    assertThat(lastExpressionInFunction(
      "global a",
      "a = 42",
      "a").type()).isEqualTo(anyType());
  }

  @Test
  void simple_propagation_between_variables() {
    assertThat(lastExpressionInFunction(
      "a = ''",
      "b = a",
      "c = b",
      "c").type()).isEqualTo(STR);
  }

  @Test
  void compound_statement_str() {
    assertThat(lastExpressionInFunction(
      "a = 'hello '",
      "b = 'world'",
      "a += b",
      "a").type()).isEqualTo(STR);
  }

  @Test
  void compound_statement_list() {
    assertThat(lastExpressionInFunction(
      "a = []",
      "b = 'world'",
      "a += b",
      "a").type()).isEqualTo(LIST);
  }

  @Test
  void compound_assignment_no_symbol() {
    assertThat(lastExpressionInFunction(
      "nonlocal x",
      "x += 10",
      "x").type()).isEqualTo(anyType());
  }

  @Test
  void reassignment() {
    assertThat(lastExpressionInFunction(
      "a = 'hello'",
      "a = 42",
      "a").type()).isEqualTo(INT);
  }

  @Test
  void variable_read_appearing_before_initialization() {
    assertThat(lastExpressionInFunction(
      "for i in range(3):",
      "  if i > 0: a = b",
      "  else:     b = 1",
      "a").type()).isEqualTo(INT);
  }

  @Test
  void cycle_between_variables_with_initialization() {
    assertThat(lastExpressionInFunction(
      "for i in range(3):",
      "  if i > 1:  b = a",
      "  elif i==1: a = b",
      "  else:      b = 1",
      "a").type()).isEqualTo(INT);
  }

  @Test
  void unresolvable_cycle_between_variables() {
    assertThat(lastExpressionInFunction(
      "if cond: a = b",
      "else:    b = a",
      "c = 1",
      "a").type()).isEqualTo(anyType());
  }

  @Test
  void unsupported_assignment() {
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
  void returned_value_type() {
    assertThat(((ReturnStatement) lastStatement("return")).returnValueType()).isEqualTo(NONE);
    assertThat(((ReturnStatement) lastStatement("return None")).returnValueType()).isEqualTo(NONE);
    assertThat(((ReturnStatement) lastStatement("return 42")).returnValueType()).isEqualTo(INT);
    assertThat(((ReturnStatement) lastStatement("return 42, 1337")).returnValueType()).isEqualTo(TUPLE);
    assertThat(((ReturnStatement) lastStatement("return (42, 1337)")).returnValueType()).isEqualTo(TUPLE);
  }

  @Test
  void random() {
    assertThat(lastExpressionInFunction(
      "x = [1,2,3]",
      "a = x.append(42)",
      "a").type()).isEqualTo(NONE);
  }

  @Test
  void imported_symbol() {
    assertThat(lastExpression(
      """
      import fcntl
      ret = fcntl.flock(..., ...)
      ret
      """
    ).type()).isEqualTo(NONE);
  }

  @Test
  void propagate_return_type_to_variable() {
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
  void multiple_types() {
    assertThat(lastExpressionInFunction(
      "if cond: a = ''",
      "else:    a = 42",
      "a").type()).isEqualTo(or(STR, INT));
  }

  @Test
  void numeric_literals() {
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
  void string_literals() {
    assertThat(lastExpression("'hello world'").type()).isEqualTo(STR);
    assertThat(lastExpression("f'hello world'").type()).isEqualTo(STR);
    assertThat(lastExpression("'hello' 'world'").type()).isEqualTo(STR);

    assertThat(lastExpression("b'hello'").type()).isEqualTo(anyType());
    assertThat(lastExpression("rb'hello'").type()).isEqualTo(anyType());

    // this throws a 'SyntaxError: cannot mix bytes and nonbytes literals'
    assertThat(lastExpression("b'hello' 'world'").type()).isEqualTo(STR);
  }

  @Test
  void list_literals() {
    assertThat(lastExpression("[]").type()).isEqualTo(LIST);
    assertThat(lastExpression("[42]").type()).isEqualTo(LIST);

    assertThat(lastExpression("[x for x in range(0, 100)]").type()).isEqualTo(LIST);
  }

  @Test
  void dict_literals() {
    assertThat(lastExpression("{}").type()).isEqualTo(DICT);
    assertThat(lastExpression("{'x' : 1, 'y' : 2}").type()).isEqualTo(DICT);

    assertThat(lastExpression("{ k:v for (k, v) in foo }").type()).isEqualTo(DICT);
  }

  @Test
  void set_literals() {
    assertThat(lastExpression("{1, 2, 3}").type()).isEqualTo(SET);

    assertThat(lastExpression("{ v for v in foo }").type()).isEqualTo(SET);
  }

  @Test
  void generator_literal() {
    assertThat(lastExpression("(v for v in foo)").type()).isEqualTo(anyType());
  }

  @Test
  void tuple_literal() {
    assertThat(lastExpression("()").type()).isEqualTo(TUPLE);
    assertThat(lastExpression("(1, 2)").type()).isEqualTo(TUPLE);
  }

  @Test
  void none_type() {
    assertThat(lastExpression("None").type()).isEqualTo(NONE);
  }

  @Test
  void true_false_literal() {
    assertThat(lastExpression("True").type()).isEqualTo(BOOL);
    assertThat(lastExpression("False").type()).isEqualTo(BOOL);
  }

  @Test
  void builtin_function_types() {
    assertThat(lastExpression("all([1, 2, 3])").type()).isEqualTo(BOOL);
    assertThat(lastExpression("round(42)").type()).isEqualTo(AnyType.ANY);
    ClassSymbol classSymbolRange = ((ClassSymbol) TypeShed.builtinSymbols().get("range"));
    assertThat(lastExpression("range(42)").type()).isEqualTo(InferredTypes.runtimeType(classSymbolRange));
    assertThat(lastExpression("getattr(42)").type()).isEqualTo(AnyType.ANY);
  }

  @Test
  void builtin_method_types() {
    assertThat(lastExpression("'abc'.capitalize()").type()).isEqualTo(STR);
    assertThat(lastExpression("list().copy()").type()).isEqualTo(LIST);
  }

  @Test
  void conditional_expressions() {
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
  void flow_sensitive_type_inference() {
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
  void flow_insensitive_when_try_except() {
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
  void nested_try_except() {
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
  void nested_try_except_2() {
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
  void execution_order_assignment_statement() {
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
  void isinstance_flow_sensitive() {
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
  void isinstance_flow_insensitive() {
    assertThat(lastExpression(
      "def f(x: int):",
      "  try:",
      "    if isinstance(x, Foo): ...",
      "  except: ...",
      "  x"
    ).type()).isEqualTo(anyType());
  }

  @Test
  void typeshed_attributes() {
    assertThat(lastExpression(
      "def f():",
      "  e = OSError()",
      "  e.errno.bit_length()"
    ).type()).isEqualTo(INT);
  }

  @Test
  void user_defined_attributes() {
    assertThat(lastExpression(
      "class Foo:",
      "  attr: int",
      "def f():",
      "  e = Foo()",
      "  e.attr.bit_length()"
    ).type()).isEqualTo(DECL_INT);
  }

  @Test
  void user_defined_attributes_list() {
    DeclaredType listOfStr = new DeclaredType(new SymbolImpl(BuiltinTypes.LIST, BuiltinTypes.LIST), List.of((DeclaredType) DECL_STR));
    assertThat(lastExpression(
      "class Foo:",
      "  attr: list[str]",
      "def f():",
      "  e = Foo()",
      "  e.attr"
    ).type()).isEqualTo(listOfStr);
  }

  @Test
  void user_defined_attributes_union() {
    DeclaredType union = new DeclaredType(new SymbolImpl("Union", "typing.Union"), List.of(new DeclaredType(BuiltinTypes.INT), new DeclaredType(BuiltinTypes.STR)));
    assertThat(lastExpression(
      "class Foo:",
      "  attr: int | str",
      "def f():",
      "  e = Foo()",
      "  e.attr"
    ).type()).isEqualTo(union);
  }

  @Test
  void user_defined_attributes_reassigned() {
    assertThat(lastExpression(
      "class Foo:",
      "  attr: int",
      "def f():",
      "  e = Foo()",
      "  e.attr = 'hello'",
      "  e.attr"
    ).type()).isEqualTo(DECL_INT);
  }

  @Test
  void child_class_method_call_is_a_member_of_parent_class() {
    ClassSymbol classA = ((ClassSymbol) ((Name) lastExpression("A", """
      class A:
        def meth(self):
          return self.foo()
      class B(A):
        def foo(self): pass
      A
      """
    )).symbol());
    assertThat(classA.canHaveMember("foo")).isTrue();
    assertThat(classA.declaredMembers()).extracting("kind", "name")
      .containsExactlyInAnyOrder(Tuple.tuple(Symbol.Kind.FUNCTION, "meth"), Tuple.tuple(Symbol.Kind.OTHER, "foo"));
  }

  @Test
  void decorators() {
    Decorator decorator = lastDecorator(
      "class A:",
      "  def dec_method():",
      "    ...",
      "my_dec = A()",
      "@my_dec.dec_method()",
      "def a_function():",
      "  ...");
    CallExpression ce = (CallExpression) decorator.expression();
    QualifiedExpression qe = (QualifiedExpression) ce.callee();
    assertThat(typeName(qe.qualifier().type())).isEqualTo("A");
    assertThat(qe.name().symbol().fullyQualifiedName()).isEqualTo("some_package.some_module.A.dec_method");
    assertThat(ce.calleeSymbol().fullyQualifiedName()).isEqualTo("some_package.some_module.A.dec_method");

    decorator = lastDecorator(
      "class A:",
      "  def __call__():",
      "    ...",
      "@A",
      "class OtherClass:",
      "  ...");
    Name name = (Name) decorator.expression();
    assertThat(name.type()).isEqualTo(TYPE);
    assertThat(name.symbol().fullyQualifiedName()).isEqualTo("some_package.some_module.A");
  }

  private static Decorator lastDecorator(String... code) {
    FileInput fileInput = parse(new SymbolTableBuilder("some_package", pythonFile("some_module")), code);
    return PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.DECORATOR));
  }
}
