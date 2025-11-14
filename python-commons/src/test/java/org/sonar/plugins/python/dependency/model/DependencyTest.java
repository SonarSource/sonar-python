/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.dependency.model;

import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static org.assertj.core.api.Assertions.assertThat;

class DependencyTest {
  public static Stream<Arguments> testNormalizationSource() {
    return Stream.of(
      Arguments.of("test", "test"),
      Arguments.of("TEST", "test"),
      Arguments.of("test---module", "test-module"),
      Arguments.of("test...module", "test-module"),
      Arguments.of("test____module", "test-module"),
      Arguments.of("test_-_--.._moduLE", "test-module")
    );
  }

  @ParameterizedTest
  @MethodSource("testNormalizationSource")
  void testNormalization(String input, String expected) {
    Dependency dependency = new Dependency(input);
    assertThat(dependency.name()).isEqualTo(expected);
  }
}
