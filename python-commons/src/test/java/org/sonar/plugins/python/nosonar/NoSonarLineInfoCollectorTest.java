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
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

class NoSonarLineInfoCollectorTest {

  @ParameterizedTest
  @MethodSource("provideCollectorParameters")
  void collector_test(String code, Map<Integer, NoSonarLineInfo> expectedLineInfos, Set<Integer> expectedEmptyNoSonarLines, String expectedSuppressedRuleIds) {
    var astNode = PythonParser.create().parse(code);

    var fileInput = new PythonTreeMaker().fileInput(astNode);

    var collector = new NoSonarLineInfoCollector();
    collector.collect("foo.py", fileInput);

    Assertions.assertThat(collector.get("foo.py")).isEqualTo(expectedLineInfos);
    Assertions.assertThat(collector.getLinesWithEmptyNoSonar("foo.py")).isEqualTo(expectedEmptyNoSonarLines);
  }

  private static Stream<Arguments> provideCollectorParameters() {
    return Stream.of(
      Arguments.of("""
          a = 1 # NOSONAR(something)
          """,
        Map.of(1, new NoSonarLineInfo(1, Set.of("something"))),
        Set.of(),
        "something"
      ),
      Arguments.of("""
          a = 1 # NOSONAR(one, two) some text
          """,
        Map.of(1, new NoSonarLineInfo(1, Set.of("one", "two"))),
        Set.of(),
        "one,two"
      ),
      Arguments.of("""
          a = 1 # NOSONAR()
          """,
        Map.of(1, new NoSonarLineInfo(1, Set.of())),
        Set.of(1),
        ""
      ),
      Arguments.of("""
          a = 1 # NOSONAR(something,)
          """,
        Map.of(1, new NoSonarLineInfo(1, Set.of("something"))),
        Set.of(),
        "something"
      ),
      Arguments.of("""
          a = 1 # NOSONAR
          """,
        Map.of(1, new NoSonarLineInfo(1, Set.of())),
        Set.of(1),
        ""
      ),
      Arguments.of("""
          a = 1 # NOSONAR some text
          """,
        Map.of(1, new NoSonarLineInfo(1, Set.of())),
        Set.of(1),
        ""
      ),
      Arguments.of("a = 1 # NOSONAR some text",
        Map.of(1, new NoSonarLineInfo(1, Set.of())),
        Set.of(1),
        ""
      ),
      Arguments.of("# NOSONAR some text",
        Map.of(1, new NoSonarLineInfo(1, Set.of())),
        Set.of(1),
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
          1, new NoSonarLineInfo(1, Set.of()),
          2, new NoSonarLineInfo(2, Set.of()),
          3, new NoSonarLineInfo(3, Set.of()),
          4, new NoSonarLineInfo(4, Set.of()),
          5, new NoSonarLineInfo(5, Set.of())
        ),
        Set.of(1, 2, 3, 4, 5),
        ""
      )
    );
  }

}
