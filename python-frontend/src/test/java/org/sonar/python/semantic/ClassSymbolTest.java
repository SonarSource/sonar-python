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
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;

public class ClassSymbolTest {

  @Test
  public void no_parents() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass");
    assertThat(fileInput.globalVariables()).hasSize(1);
  }

  @Test
  public void local_parent() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "class B(C): ",
      "  pass");
    ClassDef parentClass = (ClassDef) fileInput.statements().statements().get(0);
    Symbol parentSymbol = parentClass.name().symbol();
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.parents()).hasSize(1);
    assertThat(classSymbol.parents()).containsExactlyInAnyOrder(parentSymbol);

    assertThat(fileInput.globalVariables()).hasSize(2);
    assertThat(fileInput.globalVariables()).containsExactlyInAnyOrder(symbol, parentSymbol);
  }

  @Test
  public void multiple_local_parents() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "class A:",
      "  pass",
      "class B(C, A): ",
      "  pass");
    ClassDef parentClass = (ClassDef) fileInput.statements().statements().get(0);
    Symbol parentSymbol = parentClass.name().symbol();
    ClassDef parentClass2 = (ClassDef) fileInput.statements().statements().get(1);
    Symbol parentSymbol2 = parentClass2.name().symbol();
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(2);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.parents()).hasSize(2);
    assertThat(classSymbol.parents()).containsExactlyInAnyOrder(parentSymbol, parentSymbol2);

    assertThat(fileInput.globalVariables()).hasSize(3);
    assertThat(fileInput.globalVariables()).containsExactlyInAnyOrder(symbol, parentSymbol, parentSymbol2);
  }

  @Test
  public void unknown_parent() {
    FileInput fileInput = parse(
      "class B(C): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(0);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.parents()).hasSize(0);
    assertThat(classSymbol.hasUnresolvedParents()).isTrue();
  }

  @Test
  public void builtin_parent() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "class B(C, BaseException): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.parents()).hasSize(2);
    assertThat(classSymbol.hasUnresolvedParents()).isFalse();
  }

  @Test
  public void builtin_parent_with_unknown() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "class B(C, BaseException, unknown): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.parents()).hasSize(2);
    assertThat(classSymbol.hasUnresolvedParents()).isTrue();
  }

  @Test
  public void multiple_bindings() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "C = \"hello\"");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(0);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.OTHER)).isTrue();
  }

  @Test
  public void multiple_bindings_2() {
    FileInput fileInput = parse(
      "C = \"hello\"",
      "class C: ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isFalse();
    assertThat(Symbol.Kind.CLASS.equals(symbol.kind())).isFalse();
  }

  @Test
  public void call_expression_argument() {
    FileInput fileInput = parse(
      "def foo():",
      "  pass",
      "class C(foo()): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.parents()).hasSize(0);
    assertThat(classSymbol.hasUnresolvedParents()).isTrue();
  }

  @Test
  public void unpacking_expression_as_parent() {
    FileInput fileInput = parse(
      "foo = (Something, SomethingElse)",
      "class C(*foo): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(1);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.parents()).hasSize(0);
    assertThat(classSymbol.hasUnresolvedParents()).isTrue();
  }

  @Test
  public void parent_has_multiple_bindings() {
    FileInput fileInput = parse(
      "class C: ",
      "  pass",
      "C = \"hello\"",
      "class B(C): ",
      "  pass");
    ClassDef classDef = (ClassDef) fileInput.statements().statements().get(2);
    Symbol symbol = classDef.name().symbol();
    assertThat(symbol instanceof ClassSymbol).isTrue();
    assertThat(symbol.kind().equals(Symbol.Kind.CLASS)).isTrue();
    ClassSymbol classSymbol = (ClassSymbol) symbol;
    assertThat(classSymbol.hasUnresolvedParents()).isTrue();
  }
}
