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
        new NoSonarLineInfo(Set.of("one", "two"),"some text")
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
        new NoSonarLineInfo(Set.of("something"),"abc")
      ),
      Arguments.of(
        "# NOSONAR",
        new NoSonarLineInfo(Set.of(),"")
      ),
      Arguments.of(
        "# NOSONAR a very long comment that I don't want to keep that long because there is more than 50 characters",
        new NoSonarLineInfo(Set.of(),"a very long comment that I don't want to keep tha")
      ),
      Arguments.of(
        "# NOSONAR some text",
        new NoSonarLineInfo(Set.of(), "some text")
      ),
      Arguments.of(
        "# NOSONAR(a) # NOSONAR(b,c) # Some text # NOSONAR(d,)",
        new NoSonarLineInfo(Set.of("a", "b", "c", "d"), "# Some text")
      ),
      Arguments.of(
        "# NOSONAR(a) # NOSONAR(b,c) # Some text # NOSONAR",
        new NoSonarLineInfo(Set.of())
      ),
      Arguments.of(
        "# some text # NOSONAR(a)",
        new NoSonarLineInfo(Set.of("a"), "# some text")
      ),
      Arguments.of(
        "# NOSONAR(a) # NOSONAR(b,c) # Some text # NOSONAR",
        new NoSonarLineInfo(Set.of())
      ),
      Arguments.of(
        "# noqa: some text",
        new NoSonarLineInfo(Set.of("some text"))
      ),
      Arguments.of(
        "# noqa: a,b",
        new NoSonarLineInfo(Set.of("a", "b"))
      ),
      Arguments.of(
        "# noqa: a, b",
        new NoSonarLineInfo(Set.of("a", "b"))
      ),
      Arguments.of(
        "# noqa: a, b # noqa: c # some text # noqa: d,e",
        new NoSonarLineInfo(Set.of("a", "b", "c", "d", "e"), "# some text")
      ),
      Arguments.of(
        "# noqa: a, b # noqa: c # some text # noqa",
        new NoSonarLineInfo(Set.of())
      ),
      Arguments.of(
        "# noqa some text",
        new NoSonarLineInfo(Set.of())
      )
    );
  }

  @ParameterizedTest
  @MethodSource("provideValidationParameters")
  void validationTest(String commentString, boolean expectedIsInvalid) {
    var parser = new NoSonarInfoParser();
    var isInvalid = parser.isInvalidIssueSuppressionComment(commentString);

    Assertions.assertThat(isInvalid).isEqualTo(expectedIsInvalid);
  }

  private static Stream<Arguments> provideValidationParameters() {
    return Stream.of(
      Arguments.of("# NOSONAR", false),
      Arguments.of("# NOSONAR()", false),
      Arguments.of("# NOSONAR(a, b)", false),
      Arguments.of("# NOSONAR with some text", false),
      Arguments.of("# NOSONAR() with some text", false),
      Arguments.of("# NOSONAR(a, b) with some text", false),
      Arguments.of("# NOSONAR ()", false),
      Arguments.of("# NOSONARa", false),
      Arguments.of("# noqa", false),
      Arguments.of("# noqa: one,two", false),
      Arguments.of("# noqa:one,two", false),
      Arguments.of("# noqa- one,two", false),

      Arguments.of("# something unrelated", false),

      Arguments.of("# NOSONAR(", true),
      Arguments.of("# NOSONAR)", true),
      Arguments.of("# NOSONAR)(", true),
      Arguments.of("# NOSONAR(,)", true),
      Arguments.of("# NOSONAR(a,)", true),
      Arguments.of("# NOSONAR(a (b))", true),
      Arguments.of("# noqa: one,", true),
      Arguments.of("# noqa: ,two", true),
      Arguments.of("# noqa: , ", true),
      Arguments.of("# noqa: one,two some text", true),
      Arguments.of("# noqa: one some text", true)
    );
  }

}
