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
package org.sonar.python.semantic;

import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parse;


public class SymbolImplTest {

  @Test
  public void assert_is() {
    Symbol x = symbols("x = 42").get("x");
    assertThat(x.is(Symbol.Kind.OTHER)).isTrue();

    Symbol foo = symbols("def foo(): ...").get("foo");
    assertThat(foo.is(Symbol.Kind.FUNCTION)).isTrue();
    assertThat(foo.is(Symbol.Kind.OTHER)).isFalse();
    assertThat(foo.is(Symbol.Kind.OTHER, Symbol.Kind.FUNCTION)).isTrue();
  }

  private Map<String, Symbol> symbols(String... code) {
    FileInput fileInput = parse(new SymbolTableBuilder("", PythonTestUtils.pythonFile("foo")), code);
    return fileInput.globalVariables().stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
  }
}
