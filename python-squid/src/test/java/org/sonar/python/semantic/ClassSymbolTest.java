/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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

import org.junit.Test;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.tuple;


public class ClassSymbolTest {
  @Test
  public void no_field() {
    ClassDef empty = parseClass(
      "class C: ",
      "  pass");
    assertThat(empty.classFields()).isEmpty();
    assertThat(empty.instanceFields()).isEmpty();

    ClassDef empty2 = parseClass(
      "class C:",
      "  def f(): pass");
    assertThat(empty2.classFields()).isEmpty();
    assertThat(empty2.instanceFields()).isEmpty();
  }

  @Test
  public void class_fields() {
    ClassDef c = parseClass(
      "class C: ",
      "  f1 = 1",
      "  f1 = 2",
      "  f2 = 3");
    assertThat(c.classFields()).extracting(Symbol::name).containsExactlyInAnyOrder("f1", "f2");
    assertThat(c.instanceFields()).isEmpty();
  }

  @Test
  public void instance_fields() {
    ClassDef c1 = parseClass(
      "class C: ",
      "  def f(self):",
      "    self.a = 1",
      "    self.b = 2",
      "    x = 2",
      "  def g(self):",
      "    self.a = 3",
      "    self.c = 4");
    assertThat(c1.classFields()).isEmpty();
    assertThat(c1.instanceFields()).extracting(Symbol::name).containsExactlyInAnyOrder("a", "b", "c");

    ClassDef c2 = parseClass(
      "class C:",
      "  def f(self):",
      "     print(self.a)",
      "  def g(self):",
      "     self.a = 1");
    assertThat(c2.classFields()).isEmpty();
    assertThat(c2.instanceFields()).extracting(Symbol::name).containsExactlyInAnyOrder("a");
    Symbol field = c2.instanceFields().iterator().next();
    assertThat(field.usages())
      .extracting(Usage::kind, u -> u.tree().firstToken().line())
      .containsExactlyInAnyOrder(
        tuple(Usage.Kind.OTHER, 3),
        tuple(Usage.Kind.ASSIGNMENT_LHS, 5));
  }

  private ClassDef parseClass(String... lines) {
    FileInput fileInput = PythonTestUtils.parse(lines);
    return PythonTestUtils.getFirstChild(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
  }

}
