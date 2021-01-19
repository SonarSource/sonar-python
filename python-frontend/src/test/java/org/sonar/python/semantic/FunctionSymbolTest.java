/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.semantic;

import java.util.List;
import org.junit.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.functionSymbol;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.semantic.SymbolUtils.pathOf;

public class FunctionSymbolTest {

  @Test
  public void arity() {
    FunctionSymbol functionSymbol = functionSymbol("def fn(): pass");
    assertThat(functionSymbol.isAsynchronous()).isFalse();
    assertThat(functionSymbol.parameters()).isEmpty();

    functionSymbol = functionSymbol("async def fn(p1, p2, p3): pass");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::name).containsExactly("p1", "p2", "p3");
    assertThat(functionSymbol.hasVariadicParameter()).isFalse();
    assertThat(functionSymbol.isInstanceMethod()).isFalse();
    assertThat(functionSymbol.isAsynchronous()).isTrue();
    assertThat(functionSymbol.hasDecorators()).isFalse();
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::hasDefaultValue).containsExactly(false, false, false);
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isKeywordOnly).containsExactly(false, false, false);

    functionSymbol = functionSymbol("def fn(p1, *, p2): pass");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::name).containsExactly("p1", "p2");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::hasDefaultValue).containsExactly(false, false);
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isKeywordOnly).containsExactly(false, true);

    functionSymbol = functionSymbol("def fn(p1, /, p2): pass");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::name).containsExactly("p1", "p2");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isKeywordOnly).containsExactly(false, false);
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isPositionalOnly).containsExactly(true, false);

    functionSymbol = functionSymbol("def fn(p1, /, p2, *, p3): pass");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::name).containsExactly("p1", "p2", "p3");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isKeywordOnly).containsExactly(false, false, true);
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isPositionalOnly).containsExactly(true, false, false);

    functionSymbol = functionSymbol("def fn(p1, /, p2, *p3, p4 = False): pass");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::name).containsExactly("p1", "p2", "p3", "p4");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isKeywordOnly).containsExactly(false, false, false, true);
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isPositionalOnly).containsExactly(true, false, false, false);
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isVariadic).containsExactly(false, false, true, false);

    functionSymbol = functionSymbol("def fn(p1, p2=42): pass");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::name).containsExactly("p1", "p2");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::hasDefaultValue).containsExactly(false, true);
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isKeywordOnly).containsExactly(false, false);

    functionSymbol = functionSymbol("def fn(p1, *, p2=42): pass");
    assertThat(functionSymbol.hasVariadicParameter()).isFalse();
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::name).containsExactly("p1", "p2");
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::hasDefaultValue).containsExactly(false, true);
    assertThat(functionSymbol.parameters()).extracting(FunctionSymbol.Parameter::isKeywordOnly).containsExactly(false, true);

    functionSymbol = functionSymbol("def fn((p1,p2,p3)): pass");
    assertThat(functionSymbol.parameters()).hasSize(1);
    assertThat(functionSymbol.parameters().get(0).name()).isNull();
    assertThat(functionSymbol.parameters().get(0).hasDefaultValue()).isFalse();
    assertThat(functionSymbol.parameters().get(0).isKeywordOnly()).isFalse();
    assertThat(functionSymbol.parameters().get(0).isVariadic()).isFalse();

    functionSymbol = functionSymbol("def fn(p1: int): pass");
    assertThat(functionSymbol.parameters().get(0).declaredType().canBeOrExtend("int")).isTrue();

    functionSymbol = functionSymbol("def fn(**kwargs): pass");
    assertThat(functionSymbol.parameters()).hasSize(1);
    assertThat(functionSymbol.hasVariadicParameter()).isTrue();
    assertThat(functionSymbol.parameters().get(0).name()).isEqualTo("kwargs");
    assertThat(functionSymbol.parameters().get(0).hasDefaultValue()).isFalse();
    assertThat(functionSymbol.parameters().get(0).isKeywordOnly()).isFalse();
    assertThat(functionSymbol.parameters().get(0).isVariadic()).isTrue();

    functionSymbol = functionSymbol("def fn(p1, *args): pass");
    assertThat(functionSymbol.hasVariadicParameter()).isTrue();

    functionSymbol = functionSymbol("@something\ndef fn(p1, *args): pass");
    assertThat(functionSymbol.hasDecorators()).isTrue();
    List<String> decorators = functionSymbol.decorators();
    assertThat(decorators).hasSize(1);
    assertThat(decorators.get(0)).isEqualTo("something");

    functionSymbol = functionSymbol("@something[\"else\"]\ndef fn(p1, *args): pass");
    assertThat(functionSymbol.hasDecorators()).isTrue();
    decorators = functionSymbol.decorators();
    assertThat(decorators).isEmpty();
    assertThat(functionSymbol.isInstanceMethod()).isFalse();

    functionSymbol = functionSymbol("class A:\n  def method(self, p1): pass");
    assertThat(functionSymbol.isInstanceMethod()).isTrue();

    functionSymbol = functionSymbol("class A:\n  def method(*args, p1): pass");
    assertThat(functionSymbol.isInstanceMethod()).isTrue();

    functionSymbol = functionSymbol("class A:\n  @staticmethod\n  def method((a, b), c): pass");
    assertThat(functionSymbol.isInstanceMethod()).isFalse();

    functionSymbol = functionSymbol("class A:\n  @staticmethod\n  def method(p1, p2): pass");
    assertThat(functionSymbol.isInstanceMethod()).isFalse();

    functionSymbol = functionSymbol("class A:\n  @classmethod\n  def method(self, p1): pass");
    assertThat(functionSymbol.isInstanceMethod()).isFalse();
    assertThat(functionSymbol.hasDecorators()).isTrue();

    functionSymbol = functionSymbol("class A:\n  @dec\n  def method(self, p1): pass");
    assertThat(functionSymbol.isInstanceMethod()).isTrue();
    assertThat(functionSymbol.hasDecorators()).isTrue();

    functionSymbol = functionSymbol("class A:\n  @some[\"thing\"]\n  def method(self, p1): pass");
    assertThat(functionSymbol.isInstanceMethod()).isTrue();
    assertThat(functionSymbol.hasDecorators()).isTrue();
  }

  @Test
  public void reassigned_symbol() {
    FileInput tree = parse(
      "def fn(): pass",
      "fn = 42"
    );
    FunctionDef functionDef = (FunctionDef) tree.statements().statements().get(0);
    Symbol symbol = functionDef.name().symbol();
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.AMBIGUOUS);

    tree = parse(
      "fn = 42",
      "def fn(): pass"
    );
    functionDef = (FunctionDef) tree.statements().statements().get(1);
    symbol = functionDef.name().symbol();
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.AMBIGUOUS);

    tree = parse(
      "def fn(p1, p2): pass",
      "def fn(): pass"
    );
    functionDef = (FunctionDef) tree.statements().statements().get(0);
    symbol = functionDef.name().symbol();
    assertThat(symbol.kind()).isEqualTo(Symbol.Kind.AMBIGUOUS);
  }

  @Test
  public void owner() {
    FileInput fileInput = PythonTestUtils.parse(
      "class A:",
      "  def foo(self): pass"
    );
    ClassDef classDef = PythonTestUtils.getFirstDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
    FunctionDef funcDef = PythonTestUtils.getFirstDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF));
    FunctionSymbolImpl functionSymbol = ((FunctionSymbolImpl) funcDef.name().symbol());
    assertThat(functionSymbol.owner()).isEqualTo(classDef.name().symbol());

    fileInput = PythonTestUtils.parse(
      "def foo(): pass"
    );
    funcDef = PythonTestUtils.getFirstDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF));
    functionSymbol = ((FunctionSymbolImpl) funcDef.name().symbol());
    assertThat(functionSymbol.owner()).isNull();
  }

  @Test
  public void instance_method() {
    FileInput fileInput = PythonTestUtils.parse(
      "class A:",
      "  def foo(self): pass",
      "  def __new__(cls, a): pass",
      "  @staticmethod",
      "  def static_foo(): pass",
      "  @classmethod",
      "  def class_foo(): pass"
    );
    ClassSymbol classSymbol = TreeUtils.getClassSymbolFromDef(((ClassDef) fileInput.statements().statements().get(0)));
    FunctionSymbol foo = (FunctionSymbol) classSymbol.resolveMember("foo").get();
    assertThat(foo.isInstanceMethod()).isTrue();

    FunctionSymbol static_foo = (FunctionSymbol) classSymbol.resolveMember("static_foo").get();
    assertThat(static_foo.isInstanceMethod()).isFalse();

    FunctionSymbol class_foo = (FunctionSymbol) classSymbol.resolveMember("class_foo").get();
    assertThat(class_foo.isInstanceMethod()).isFalse();

    FunctionSymbol newMethod = (FunctionSymbol) classSymbol.resolveMember("__new__").get();
    assertThat(newMethod.isInstanceMethod()).isFalse();
  }

  @Test
  public void locations() {
    PythonFile foo = PythonTestUtils.pythonFile("foo");

    FunctionSymbol functionSymbol = functionSymbol(foo, "def foo(param1, param2): ...");
    assertThat(functionSymbol.parameters().get(0).location()).isEqualToComparingFieldByField(new LocationInFile(pathOf(foo).toString(), 1, 8, 1, 14));
    assertThat(functionSymbol.parameters().get(1).location()).isEqualToComparingFieldByField(new LocationInFile(pathOf(foo).toString(), 1, 16, 1, 22));
    assertThat(functionSymbol.definitionLocation()).isEqualToComparingFieldByField(new LocationInFile(pathOf(foo).toString(), 1, 4, 1, 7));

    functionSymbol = functionSymbol(foo, "def foo(*param1): ...");
    assertThat(functionSymbol.parameters().get(0).location()).isEqualToComparingFieldByField(new LocationInFile(pathOf(foo).toString(), 1, 8, 1, 15));

    functionSymbol = functionSymbol(foo, "def foo((a, b)): ...");
    assertThat(functionSymbol.parameters().get(0).location()).isEqualToComparingFieldByField(new LocationInFile(pathOf(foo).toString(), 1, 8, 1, 14));

    FileInput fileInput = parse(new SymbolTableBuilder(foo), "all([1,2,3])");
    CallExpression callExpression = (CallExpression) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.CALL_EXPR)).get(0);
    FunctionSymbol builtinFunctionSymbol = (FunctionSymbol) callExpression.calleeSymbol();
    assertThat(builtinFunctionSymbol.definitionLocation()).isNull();
    assertThat(builtinFunctionSymbol.parameters().get(0).location()).isNull();
  }

  @Test
  public void declared_return_type() {
    FunctionSymbolImpl functionSymbol = (FunctionSymbolImpl) functionSymbol("def foo() -> int: ...");
    assertThat(functionSymbol.declaredReturnType().canBeOrExtend("int")).isTrue();
    assertThat(InferredTypes.typeName(functionSymbol.declaredReturnType())).isEqualTo("int");

    functionSymbol = (FunctionSymbolImpl) functionSymbol(
      "from typing import Optional",
      "def foo() -> Optional[int]: ..."
    );
    assertThat(functionSymbol.declaredReturnType().canBeOrExtend("int")).isTrue();
    assertThat(InferredTypes.typeName(functionSymbol.declaredReturnType())).isEqualTo("Optional[int]");
  }
}
