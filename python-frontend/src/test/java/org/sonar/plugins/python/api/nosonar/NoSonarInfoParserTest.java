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
package org.sonar.plugins.python.api.nosonar;

import java.util.Set;
import java.util.stream.Stream;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class NoSonarInfoParserTest {

  @Test
  void noNoSonarCommentTest() {
    var parser = new NoSonarInfoParser();
    var parsingResult = parser.parse("# comment string");
    Assertions.assertThat(parsingResult).isNotPresent();
  }

  @ParameterizedTest
  @MethodSource("provideParserParameters")
  void parsingTest(String commentString, NoSonarLineInfo expectedLineInfo) {
    var parser = new NoSonarInfoParser();
    var parsingResult = parser.parse(commentString);

    Assertions.assertThat(parsingResult).isPresent().contains(expectedLineInfo);
  }

  private static Stream<Arguments> provideParserParameters() {
    return Stream.of(
      Arguments.of(
        "# NOSONAR(something)",
        new NoSonarLineInfo(Set.of("something")
        )
      ),
      Arguments.of(
        "# NOSONAR(one, two) some text",
        new NoSonarLineInfo(Set.of("one", "two"))
      ),
      Arguments.of(
        "# NOSONAR()",
        new NoSonarLineInfo(Set.of())
      ),
      Arguments.of(
        "# NOSONAR(something,)",
        new NoSonarLineInfo(Set.of("something"))
      ),
      Arguments.of(
        "# NOSONAR(something,) abc",
        new NoSonarLineInfo(Set.of("something"))
      ),
      Arguments.of(
        "# NOSONAR",
        new NoSonarLineInfo(Set.of())
      ),
      Arguments.of(
        "# NOSONAR some text",
        new NoSonarLineInfo(Set.of())
      ),
      Arguments.of(
        "# noqa: some text",
        new NoSonarLineInfo(Set.of("some"))
      ),
      Arguments.of(
        "# noqa: a,b",
        new NoSonarLineInfo(Set.of("a", "b"))
      ),
      Arguments.of(
        "# noqa: a, b",
        new NoSonarLineInfo(Set.of("a"))
      )
    );
  }

}
