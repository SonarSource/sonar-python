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
package org.sonar.python.index;


import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.semantic.v2.converter.PythonTypeToDescriptorConverter;
import org.sonar.plugins.python.api.types.v2.PythonType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.index.DescriptorToProtobufTestUtils.assertDescriptorToProtobuf;
import static org.sonar.python.index.DescriptorsToProtobuf.fromProtobuf;
import static org.sonar.python.index.DescriptorsToProtobuf.toProtobuf;
import static org.sonar.python.types.v2.TypesTestUtils.lastName;

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
    Name name = lastName(code);
    PythonType pythonType = name.typeV2();
    SymbolV2 symbol = name.symbolV2();
    PythonTypeToDescriptorConverter converter = new PythonTypeToDescriptorConverter();
    VariableDescriptor variableDescriptor = (VariableDescriptor) converter.convert("my_package.mod", symbol, Set.of(pythonType));
    assertThat(variableDescriptor.kind()).isEqualTo(Descriptor.Kind.VARIABLE);
    assertThat(variableDescriptor.name()).isEqualTo(symbol.name());
    assertThat(variableDescriptor.fullyQualifiedName()).isEqualTo("my_package.mod." + symbol.name());
    return variableDescriptor;
  }

  void assertVariableDescriptors(VariableDescriptor first, VariableDescriptor second) {
    assertThat(first.name()).isEqualTo(second.name());
    assertThat(first.fullyQualifiedName()).isEqualTo(second.fullyQualifiedName());
    assertThat(first.annotatedType()).isEqualTo(second.annotatedType());
  }
}
