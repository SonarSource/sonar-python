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
package org.sonar.python.api;

import com.sonar.sslr.api.AstNode;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class PythonTokenTypeTest {

  @Test
  void test() {
    assertThat(PythonTokenType.values()).hasSize(9);

    AstNode astNode = mock(AstNode.class);
    for (PythonTokenType tokenType : PythonTokenType.values()) {
      assertThat(tokenType.getName()).isEqualTo(tokenType.name());
      assertThat(tokenType.getValue()).isEqualTo(tokenType.name());
      assertThat(tokenType.hasToBeSkippedFromAst(astNode)).isFalse();
    }
  }

}
