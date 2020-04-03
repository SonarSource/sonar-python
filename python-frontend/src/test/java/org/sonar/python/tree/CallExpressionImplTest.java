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
package org.sonar.python.tree;

import java.util.Iterator;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.getLastDescendant;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.PythonTestUtils.parse;

public class CallExpressionImplTest {

  @Test
  public void constructor_type() {
    FileInput fileInput = parse(
      "class A: ...",
      "A()");

    ClassDef classDef = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
    CallExpression call = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CALL_EXPR));

    assertThat(call.type()).isEqualTo(InferredTypes.runtimeType(classDef.name().symbol()));
  }

  @Test
  public void function_call_type() {
    assertThat(lastExpression("len()").type()).isEqualTo(InferredTypes.INT);
  }

  @Test
  public void ambiguous_callee_symbol_type() {
    FileInput fileInput = parse(
      "class A: ...",
      "class A: ...",
      "A()");

    ClassDef classDef = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
    CallExpression call = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CALL_EXPR));

    AmbiguousSymbol ambiguousSymbol = (AmbiguousSymbol) classDef.name().symbol();
    Iterator<Symbol> iterator = ambiguousSymbol.alternatives().iterator();
    InferredType firstClassType = InferredTypes.runtimeType(iterator.next());
    InferredType secondClassType = InferredTypes.runtimeType(iterator.next());
    assertThat(call.type()).isEqualTo(InferredTypes.or(firstClassType, secondClassType));
    assertThat(call.type()).isEqualTo(firstClassType);
    assertThat(call.type()).isEqualTo(secondClassType);

    fileInput = parse(
      "class Base: ...",
      "class A(Base): ...",
      "class A: ...",
      "A()");

    classDef = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
    call = getLastDescendant(fileInput, t -> t.is(Tree.Kind.CALL_EXPR));

    ambiguousSymbol = (AmbiguousSymbol) classDef.name().symbol();
    iterator = ambiguousSymbol.alternatives().iterator();
    firstClassType = InferredTypes.runtimeType(iterator.next());
    secondClassType = InferredTypes.runtimeType(iterator.next());
    assertThat(call.type()).isEqualTo(InferredTypes.or(firstClassType, secondClassType));
    assertThat(call.type()).isNotEqualTo(firstClassType);
    assertThat(call.type()).isNotEqualTo(secondClassType);
  }

  @Test
  public void not_callable_callee_symbol_type() {
    assertThat(lastExpression(
      "x = 42",
      "x()"
    ).type()).isEqualTo(InferredTypes.anyType());
  }

  @Test
  public void null_callee_symbol_type() {
    assertThat(lastExpression("x()").type()).isEqualTo(InferredTypes.anyType());
  }
}
