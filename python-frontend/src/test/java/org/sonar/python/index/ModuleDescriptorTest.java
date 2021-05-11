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


import java.util.Collection;
import java.util.Map;
import org.junit.Test;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.PythonTestUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;

public class ModuleDescriptorTest {


  @Test
  public void module() {
    PythonFile mod = PythonTestUtils.pythonFile("mod");
    FileInput fileInput = parseWithoutSymbols(
      "class A: ...",
      "def foo(): ...",
      "bar: int = 42"
    );
    ProjectDescriptor projectDescriptor = new ProjectDescriptor();
    projectDescriptor.addModule(fileInput, "package", mod);
    Map<String, ModuleDescriptor> modules = projectDescriptor.modules();
    assertThat(modules).hasSize(1);
    ModuleDescriptor moduleDescriptor = modules.get("package.mod");
    assertThat(moduleDescriptor.kind()).isEqualTo(Descriptor.Kind.MODULE);
    assertThat(moduleDescriptor.name()).isEqualTo("mod");
    assertThat(moduleDescriptor.fullyQualifiedName()).isEqualTo("package.mod");
    assertThat(moduleDescriptor.classes()).extracting(ClassDescriptor::fullyQualifiedName).containsExactly("package.mod.A");
    assertThat(moduleDescriptor.functions()).extracting(FunctionDescriptor::fullyQualifiedName).containsExactly("package.mod.foo");
    assertThat(moduleDescriptor.variables()).extracting(VariableDescriptor::fullyQualifiedName).containsExactly("package.mod.bar");

    Descriptor classA = moduleDescriptor.descriptorsWithFQN("package.mod.A").iterator().next();
    assertThat(classA.fullyQualifiedName()).isEqualTo("package.mod.A");
  }

  @Test
  public void ambiguousFQN() {
    PythonFile mod = PythonTestUtils.pythonFile("mod");
    FileInput fileInput = parseWithoutSymbols(
      "class A: ...",
      "def A(): ..."
    );
    ProjectDescriptor projectDescriptor = new ProjectDescriptor();
    projectDescriptor.addModule(fileInput, "package", mod);
    Map<String, ModuleDescriptor> modules = projectDescriptor.modules();
    ModuleDescriptor moduleDescriptor = modules.get("package.mod");

    Collection<Descriptor> descriptors = moduleDescriptor.descriptorsWithFQN("package.mod.A");
    assertThat(descriptors).hasSize(2);
    assertThat(descriptors).extracting(Descriptor::kind).containsExactlyInAnyOrder(Descriptor.Kind.CLASS, Descriptor.Kind.FUNCTION);
  }
}
