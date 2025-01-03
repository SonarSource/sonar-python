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
package org.sonar.python.semantic;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;


class UsageImplTest {

  @Test
  void binding_usages() {
    assertBindingUsage(Usage.Kind.ASSIGNMENT_LHS, true);
    assertBindingUsage(Usage.Kind.COMPOUND_ASSIGNMENT_LHS, true);
    assertBindingUsage(Usage.Kind.IMPORT, true);
    assertBindingUsage(Usage.Kind.LOOP_DECLARATION, true);
    assertBindingUsage(Usage.Kind.COMP_DECLARATION, true);
    assertBindingUsage(Usage.Kind.PARAMETER, true);
    assertBindingUsage(Usage.Kind.PATTERN_DECLARATION, true);
    assertBindingUsage(Usage.Kind.OTHER, false);
  }

  private static void assertBindingUsage(Usage.Kind kind, boolean isBinding) {
    Tree mockTree = Mockito.mock(Tree.class);
    UsageImpl usage = new UsageImpl(mockTree, kind);
    assertThat(usage.kind()).isEqualTo(kind);
    assertThat(usage.isBindingUsage()).isEqualTo(isBinding);
  }
}
