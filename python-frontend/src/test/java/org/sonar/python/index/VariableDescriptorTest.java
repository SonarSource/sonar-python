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
package org.sonar.python.index;


import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.python.tree.TreeUtils;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.lastExpression;
import static org.sonar.python.index.DescriptorToProtobufTestUtils.assertDescriptorToProtobuf;
import static org.sonar.python.index.DescriptorsToProtobuf.fromProtobuf;
import static org.sonar.python.index.DescriptorsToProtobuf.toProtobuf;
import static org.sonar.python.index.DescriptorUtils.descriptor;

class VariableDescriptorTest {

  @Test
  void variableDescriptor() {
    VariableDescriptor x = lastVariableDescriptor(
      "x: int = 42",
      "x");
    // only typeshed symbols have it != null
    assertThat(x.annotatedType()).isNull();
    assertDescriptorToProtobuf(x);
  }

  @Test
  void protobufSerializationWithAnnotatedReturnType() {
    VariableDescriptor variableDescriptor = new VariableDescriptor("x", "mod.x", "str");
    assertVariableDescriptors(variableDescriptor, fromProtobuf(toProtobuf(variableDescriptor)));
  }

  private VariableDescriptor lastVariableDescriptor(String... code) {
    Symbol symbol = TreeUtils.getSymbolFromTree(lastExpression(code)).get();

    VariableDescriptor variableDescriptor = (VariableDescriptor) descriptor(symbol);
    assertThat(variableDescriptor.kind()).isEqualTo(Descriptor.Kind.VARIABLE);
    assertThat(variableDescriptor.name()).isEqualTo(symbol.name());
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo(symbol.fullyQualifiedName());
    return variableDescriptor;
  }

  void assertVariableDescriptors(VariableDescriptor first, VariableDescriptor second) {
    assertThat(first.name()).isEqualTo(second.name());
    assertThat(first.fullyQualifiedName()).isEqualTo(second.fullyQualifiedName());
    assertThat(first.annotatedType()).isEqualTo(second.annotatedType());
  }
}
