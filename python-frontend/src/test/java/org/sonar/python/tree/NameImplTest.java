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

import org.junit.Test;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.ClassSymbolImpl;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.InferredTypes;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

public class NameImplTest {

  @Test
  public void type() {
    Token token = new TokenImpl(mock(com.sonar.sslr.api.Token.class));
    NameImpl name = new NameImpl(token, true);
    assertThat(name.type()).isEqualTo(InferredTypes.anyType());

    SymbolImpl symbol = new SymbolImpl("x", null);
    name.setSymbol(symbol);
    assertThat(name.type()).isEqualTo(InferredTypes.anyType());
    InferredType str = InferredTypes.runtimeType(new ClassSymbolImpl("str", "str"));
    symbol.setInferredType(str);
    assertThat(name.type()).isEqualTo(str);
  }
}
