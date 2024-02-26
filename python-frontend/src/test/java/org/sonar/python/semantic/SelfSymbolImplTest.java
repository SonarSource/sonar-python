/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;


class SelfSymbolImplTest {

  @Test
  void removeUsages() {
    FileInput fileInput = parse(
      "class A:",
      "  def foo(self):",
      "    self.foo = 42"
    );
    FunctionDef method = PythonTestUtils.getLastDescendant(fileInput, tree -> tree.is(Tree.Kind.FUNCDEF));
    Parameter selfParameterTree = method.parameters().nonTuple().get(0);
    Symbol selfSymbol = selfParameterTree.name().symbol();
    ((SymbolImpl) selfSymbol).removeUsages();
    assertThat(selfSymbol.usages()).isEmpty();
    ClassDef classDef = PythonTestUtils.getLastDescendant(fileInput, tree -> tree.is(Tree.Kind.CLASSDEF));
    assertThat(classDef.instanceFields()).isNotEmpty();
    assertThat(classDef.instanceFields()).allMatch(symbol -> symbol.usages().isEmpty());
  }

}
