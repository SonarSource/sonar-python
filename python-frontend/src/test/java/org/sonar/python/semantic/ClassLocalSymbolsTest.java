/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic;

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;


class ClassLocalSymbolsTest {
  @Test
  void no_field() {
    ClassDef empty = parseClass(
      "class C: ",
      "  pass");
    assertThat(empty.classFields()).isEmpty();
    assertThat(empty.instanceFields()).isEmpty();
  }

  @Test
  void only_methods() {
    ClassDef empty2 = parseClass(
      "class C:",
      "  def f(): pass");
    assertThat(empty2.classFields()).extracting(Symbol::name).containsExactly("f");
    assertThat(empty2.instanceFields()).isEmpty();
  }

  @Test
  void class_fields() {
    ClassDef c = parseClass(
      "class C: ",
      "  f1 = 1",
      "  f1 = 2",
      "  f2 = 3");
    assertThat(c.classFields()).extracting(Symbol::name).containsExactlyInAnyOrder("f1", "f2");
    assertThat(c.instanceFields()).isEmpty();
  }

  @Test
  void instance_fields() {
    ClassDef c1 = parseClass(
      "class C: ",
      "  def f(self):",
      "    self.a = 1",
      "    self.b = 2",
      "    x = 2",
      "  def g(self):",
      "    self.a = 3",
      "    self.c = 4");
    assertThat(c1.classFields()).extracting(Symbol::name).containsExactlyInAnyOrder("f", "g");
    assertThat(c1.instanceFields()).extracting(Symbol::name).containsExactlyInAnyOrder("a", "b", "c");

    ClassDef c2 = parseClass(
      "class C:",
      "  def f(self):",
      "     print(self.a)",
      "  def g(self):",
      "     self.a = 1");
    assertThat(c2.classFields()).extracting(Symbol::name).containsExactlyInAnyOrder("f", "g");
    assertThat(c2.instanceFields()).extracting(Symbol::name).containsExactlyInAnyOrder("a");
    Symbol field = c2.instanceFields().iterator().next();
    assertThat(field.usages())
      .extracting(Usage::kind, u -> u.tree().firstToken().line())
      .containsExactlyInAnyOrder(
        tuple(Usage.Kind.OTHER, 3),
        tuple(Usage.Kind.ASSIGNMENT_LHS, 5));
  }

  @Test
  void same_name() {
    ClassDef c = parseClass(
      "class C: ",
      "  f1 = 1",
      "  def fn():",
      "    self.f1 = 2");
    assertThat(c.classFields()).extracting(Symbol::name).containsExactlyInAnyOrder("f1", "fn");
    assertThat(c.instanceFields()).isEmpty();
  }

  @Test
  void same_name_method_fn() {
    FileInput fileInput = PythonTestUtils.parse(
      "def fn(): pass",
      "class C: ",
      "  class X: pass",
      "  def fn(param = X):",
      "    fn()",
      "  class Y(X): pass");

    Symbol functionFn = ((FunctionDef) fileInput.statements().statements().get(0)).name().symbol();
    ClassDef c = PythonTestUtils.getFirstChild(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
    Symbol methodFn = ((FunctionDef) c.body().statements().get(1)).name().symbol();
    Symbol classXSymbol = ((ClassDef) c.body().statements().get(0)).name().symbol();

    assertThat(functionFn.usages()).extracting(Usage::kind).containsExactlyInAnyOrder(Usage.Kind.FUNC_DECLARATION, Usage.Kind.OTHER);
    assertThat(methodFn.usages()).extracting(Usage::kind).containsExactlyInAnyOrder(Usage.Kind.FUNC_DECLARATION);
    assertThat(classXSymbol.usages()).extracting(Usage::kind).containsExactlyInAnyOrder(Usage.Kind.CLASS_DECLARATION, Usage.Kind.OTHER, Usage.Kind.OTHER);
  }

  private ClassDef parseClass(String... lines) {
    FileInput fileInput = PythonTestUtils.parse(lines);
    return PythonTestUtils.getFirstChild(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
  }

}
