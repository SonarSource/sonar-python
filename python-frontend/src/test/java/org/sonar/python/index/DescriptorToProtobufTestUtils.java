/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.index.DescriptorsToProtobuf.fromProtobuf;
import static org.sonar.python.index.DescriptorsToProtobuf.toProtobufModuleDescriptor;

class DescriptorToProtobufTestUtils {

  static void assertDescriptorToProtobuf(Descriptor descriptor) {
    // This asserts that a descriptor will be retrieved unaltered after being serialized and deserialized as a protobuf module descriptor.
    assertThat(fromProtobuf(toProtobufModuleDescriptor(Set.of(descriptor)))).usingRecursiveFieldByFieldElementComparator().containsExactlyInAnyOrderElementsOf(Set.of(descriptor));
  }
}
