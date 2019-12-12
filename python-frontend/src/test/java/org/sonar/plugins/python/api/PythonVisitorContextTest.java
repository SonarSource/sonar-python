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
package org.sonar.plugins.python.api;

import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;

public class PythonVisitorContextTest {
  @Test
  public void fullyQualifiedModuleName() {
    FileInput fileInput = PythonTestUtils.parse("def foo(): pass");
    PythonFile pythonFile = Mockito.mock(PythonFile.class, "my_module.py");
    Mockito.when(pythonFile.fileName()).thenReturn("my_module.py");
    new PythonVisitorContext(fileInput, pythonFile, null, "my_package");
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("my_package.my_module.foo");

    // no package
    new PythonVisitorContext(fileInput, pythonFile, null, "");
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("my_module.foo");

    // file without extension
    Mockito.when(pythonFile.fileName()).thenReturn("my_module");
    new PythonVisitorContext(fileInput, pythonFile, null, "my_package");
    functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("my_package.my_module.foo");
  }

  @Test
  public void initModuleFullyQualifiedName() {
    FileInput fileInput = PythonTestUtils.parse("def fn(): pass");
    PythonFile pythonFile = Mockito.mock(PythonFile.class, "__init__.py");
    Mockito.when(pythonFile.fileName()).thenReturn("__init__.py");
    new PythonVisitorContext(fileInput, pythonFile, null, "foo.bar");
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("foo.bar.fn");

    // no package
    new PythonVisitorContext(fileInput, pythonFile, null, "");
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("fn");
  }
}
