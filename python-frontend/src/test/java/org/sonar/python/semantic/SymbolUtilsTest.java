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

import java.io.File;
import java.util.Set;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FileInput;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;
import static org.sonar.python.semantic.SymbolUtils.pythonPackageName;

public class SymbolUtilsTest {

  @Test
  public void global_symbols() {
    FileInput tree = parse(
      "obj1 = 42",
      "obj2: int = 42",
      "def fn(): pass",
      "class A: pass"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "mod");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("obj1", "obj2", "fn", "A");
    assertThat(globalSymbols).extracting(Symbol::fullyQualifiedName).containsExactlyInAnyOrder("mod.obj1", "mod.obj2", "mod.fn", "mod.A");
  }

  @Test
  public void global_symbols_private_by_convention() {
    // although being private by convention, it's considered as exported
    FileInput tree = parse(
      "def _private_fn(): pass"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "mod");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("_private_fn");
  }

  @Test
  public void local_symbols_not_exported() {
    FileInput tree = parse(
      "def fn():",
      "  def inner(): pass",
      "  class Inner_class: pass",
      "class A:",
      "  def meth(): pass"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "mod");
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("fn", "A");
  }

  @Test
  public void redefined_symbols() {
    FileInput tree = parse(
      "def fn(): pass",
      "def fn(): ...",
      "if True:",
      "  conditionally_defined = 1",
      "else:",
      "  conditionally_defined = 2"
    );
    Set<Symbol> globalSymbols = SymbolUtils.globalSymbols(tree, "mod");
    // for the time being, accepting multiple symbols having the same name
    assertThat(globalSymbols).extracting(Symbol::name).containsExactlyInAnyOrder("fn", "fn", "conditionally_defined", "conditionally_defined");
  }

  @Test
  public void package_name_by_file() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/__init__.py"), baseDir)).isEqualTo("sound");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/__init__.py"), baseDir)).isEqualTo("sound.formats");
    assertThat(pythonPackageName(new File(baseDir, "packages/sound/formats/wavread.py"), baseDir)).isEqualTo("sound.formats");
  }
}
