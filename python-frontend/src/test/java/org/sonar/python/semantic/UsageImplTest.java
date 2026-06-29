/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Tree;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;


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
    Tree mockTree = mock(Tree.class);
    UsageImpl usage = new UsageImpl(mockTree, kind);
    assertThat(usage.kind()).isEqualTo(kind);
    assertThat(usage.isBindingUsage()).isEqualTo(isBinding);
  }
}
