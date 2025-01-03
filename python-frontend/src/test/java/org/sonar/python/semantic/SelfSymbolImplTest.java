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
