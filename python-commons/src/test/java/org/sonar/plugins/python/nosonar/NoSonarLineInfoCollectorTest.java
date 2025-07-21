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
package org.sonar.plugins.python.nosonar;

import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.nosonar.NoSonarLineInfo;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

class NoSonarLineInfoCollectorTest {

  @ParameterizedTest
  @MethodSource("provideCollectorParameters")
  void collector_test(
    String code,
    Map<Integer, NoSonarLineInfo> expectedLineInfos,
    Set<Integer> expectedEmptyNoSonarLines,
    String expectedSuppressedRuleIds,
    String expectedComments
  ) {
    var astNode = PythonParser.create().parse(code);

    var fileInput = new PythonTreeMaker().fileInput(astNode);

    var collector = new NoSonarLineInfoCollector();
    collector.collect("foo.py", fileInput);
    Assertions.assertThat(collector.get("foo.py")).isEqualTo(expectedLineInfos);
    Assertions.assertThat(collector.getLinesWithEmptyNoSonar("foo.py")).isEqualTo(expectedEmptyNoSonarLines);
    Assertions.assertThat(collector.getSuppressedRuleIds()).isEqualTo(expectedSuppressedRuleIds);
    Assertions.assertThat(collector.getCommentWithExactlyOneRuleSuppressed()).isEqualTo(expectedComments);
  }

  private static Stream<Arguments> provideCollectorParameters() {
    return Stream.of(
      Arguments.of("""
          a = 1 # NOSONAR(something)
          """,
        Map.of(1, new NoSonarLineInfo(Set.of("something"))),
        Set.of(),
        "something",
        "something:"
      ),
      Arguments.of("""
          a = 1 # NOSONAR(one, two) some text
          """,
        Map.of(1, new NoSonarLineInfo(Set.of("one", "two"), "some text")),
        Set.of(),
        "one,two",
        ""
      ),
      Arguments.of("""
          a = 1 # NOSONAR()
          """,
        Map.of(1, new NoSonarLineInfo(Set.of())),
        Set.of(1),
        "",
        ""
      ),
      Arguments.of("""
          a = 1 # NOSONAR(something,)
          """,
        Map.of(1, new NoSonarLineInfo(Set.of("something"))),
        Set.of(),
        "something",
        "something:"
      ),
      Arguments.of("""
          a = 1 # NOSONAR(something,) this is a comment on why I suppressed this rule
          """,
        Map.of(1, new NoSonarLineInfo(Set.of("something"), "this is a comment on why I suppressed this rule")),
        Set.of(),
        "something",
        "something:this is a comment on why I suppressed this rule"
      ),
      Arguments.of("""
          a = 1 # NOSONAR
          """,
        Map.of(1, new NoSonarLineInfo(Set.of(), "")),
        Set.of(1),
        "",
        ""
      ),
      Arguments.of("""
          a = 1 # NOSONAR some text
          """,
        Map.of(1, new NoSonarLineInfo(Set.of(), "some text")),
        Set.of(1),
        "",
        ""
      ),
      Arguments.of("""
          a = 1 # NOSONAR(aRule) this is a very long comment and I don't want to send all these characters to sonarqube through telemetry
          """,
        Map.of(1, new NoSonarLineInfo(Set.of("aRule"), "this is a very long comment and I don't want to s")),
        Set.of(),
        "aRule",
        "aRule:this is a very long comment and I don't want to s"
      ),
      Arguments.of("""
          a = 1 # NOSONAR some comment
          b = 2 # NOSONAR(bRule) b comment
          c = 3 # NOSONAR(cRule) c comment
          d, e = 4,5 # NOSONAR(dRule, eRule) d and e comment
          """,
        Map.of(
          1, new NoSonarLineInfo(Set.of(), "some comment"),
          2, new NoSonarLineInfo(Set.of("bRule"), "b comment"),
          3, new NoSonarLineInfo(Set.of("cRule"), "c comment"),
          4, new NoSonarLineInfo(Set.of("dRule", "eRule"), "d and e comment")
        ),
        Set.of(1),
        "bRule,cRule,dRule,eRule",
        "bRule:b comment;;cRule:c comment"
      ),
      Arguments.of("a = 1 # NOSONAR some text",
        Map.of(1, new NoSonarLineInfo(Set.of(), "some text")),
        Set.of(1),
        "",
        ""
      ),
      Arguments.of("# NOSONAR some text",
        Map.of(1, new NoSonarLineInfo(Set.of(), "some text")),
        Set.of(1),
        "",
        ""
      ),
      Arguments.of("# noqa: some text",
        Map.of(1, new NoSonarLineInfo(Set.of("some text"))),
        Set.of(),
        "some text",
        "some text:"
      ),
      Arguments.of("# noqa: a,b",
        Map.of(1, new NoSonarLineInfo(Set.of("a", "b"))),
        Set.of(),
        "a,b",
        ""
      ),
      Arguments.of("# noqa: a, b",
        Map.of(1, new NoSonarLineInfo(Set.of("a", "b"))),
        Set.of(),
        "a,b",
        ""
      ),
      Arguments.of("# noqa: a, b # Some text",
        Map.of(1, new NoSonarLineInfo(Set.of("a", "b"), "# Some text")),
        Set.of(),
        "a,b",
        ""
      ),
      Arguments.of("""
          ""\"
          1
          2
          3
          ""\" # NOSONAR
          """,
        Map.of(
          1, new NoSonarLineInfo(Set.of()),
          2, new NoSonarLineInfo(Set.of()),
          3, new NoSonarLineInfo(Set.of()),
          4, new NoSonarLineInfo(Set.of()),
          5, new NoSonarLineInfo(Set.of())
        ),
        Set.of(1, 2, 3, 4, 5),
        "",
        ""
      )
    );
  }

}
