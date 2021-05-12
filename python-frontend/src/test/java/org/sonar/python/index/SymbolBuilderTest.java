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
package org.sonar.python.index;


import java.util.Collections;
import org.junit.Before;
import org.junit.Test;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;

public class SymbolBuilderTest {

  ProjectDescriptor projectDescriptor;

  @Before
  public void init() {
    projectDescriptor = new ProjectDescriptor();
  }

  @Test
  public void nullSymbol() {
    Symbol symbol = new SymbolBuilder(Collections.emptyMap(), projectDescriptor).build();
    assertThat(symbol).isNull();
  }

  @Test
  public void classSymbol() {
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("A")
      .withFullyQualifiedName("foo.A")
      .build();
    ClassSymbol classSymbol = (ClassSymbol) new SymbolBuilder(Collections.emptyMap(), projectDescriptor)
      .fromDescriptors(Collections.singleton(classDescriptor))
      .build();
    assertThat(classSymbol.name()).isEqualTo("A");
    assertThat(classSymbol.fullyQualifiedName()).isEqualTo("foo.A");
  }

  @Test
  public void classSymbolWithSuperClasses() {
    ClassDescriptor classDescriptor = new ClassDescriptor.ClassDescriptorBuilder()
      .withName("A")
      .withFullyQualifiedName("foo.A")
      .withSuperClasses(Collections.singleton("foo.B"))
      .build();
    ClassSymbol classSymbol = (ClassSymbol) new SymbolBuilder(Collections.emptyMap(), projectDescriptor)
      .fromDescriptors(Collections.singleton(classDescriptor))
      .build();
    assertThat(classSymbol.name()).isEqualTo("A");
    assertThat(classSymbol.fullyQualifiedName()).isEqualTo("foo.A");
  }

  @Test
  public void withFQN() {
    FileInput fileInput = PythonTestUtils.parseWithoutSymbols("def foo(): ...");
    projectDescriptor.addModule(fileInput, "my_package", PythonTestUtils.pythonFile("my_mod"));
    Symbol symbol = new SymbolBuilder(Collections.emptyMap(), projectDescriptor)
      .fromFullyQualifiedName("my_package.my_mod.foo")
      .build();

    assertThat(symbol.fullyQualifiedName()).isEqualTo("my_package.my_mod.foo");
  }
}

