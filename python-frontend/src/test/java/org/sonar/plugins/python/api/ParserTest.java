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
package org.sonar.plugins.python.api;

import com.sonar.sslr.api.RecognitionException;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.tree.FileInput;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ParserTest {

  @Test
  void test() {
    Parser parser = new Parser();

    FileInput result = parser.parse("");
    assertThat(result.statements()).isNull();

    result = parser.parse("foo()");
    assertThat(result.statements().statements()).hasSize(1);
  }

  @Test
  void parsing_error() {
    Parser parser = new Parser();
    assertThatThrownBy(() -> parser.parse("foo(")).isInstanceOf(RecognitionException.class);
  }
}
