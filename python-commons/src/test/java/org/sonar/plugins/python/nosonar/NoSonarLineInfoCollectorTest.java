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
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

class NoSonarLineInfoCollectorTest {


  @Test
  void single_parameter_nosonar_test() {
    var astNode = PythonParser.create().parse("""
      a = 1 # NOSONAR(something)
      """);

    var fileInput = new PythonTreeMaker().fileInput(astNode);

    var collector = new NoSonarLineInfoCollector();
    collector.collect("foo.py", fileInput);
    Assertions.assertThat(collector.get("foo.py"))
      .hasSize(1)
      .containsKey(1)
      .containsValue(new NoSonarLineInfo(1, Set.of("something")));
  }

  @Test
  void two_parameters_nosonar_test() {
    var astNode = PythonParser.create().parse("""
      a = 1 # NOSONAR(one, two)
      """);

    var fileInput = new PythonTreeMaker().fileInput(astNode);

    var collector = new NoSonarLineInfoCollector();
    collector.collect("foo.py", fileInput);
    Assertions.assertThat(collector.get("foo.py"))
      .hasSize(1)
      .containsKey(1)
      .containsValue(new NoSonarLineInfo(1, Set.of("one", "two")));

    Assertions.assertThat(collector.getLinesWithEmptyNoSonar("foo.py")).isEmpty();
  }

  @Test
  void empty_parameters_nosonar_test() {
    var astNode = PythonParser.create().parse("""
      a = 1 # NOSONAR()
      """);

    var fileInput = new PythonTreeMaker().fileInput(astNode);

    var collector = new NoSonarLineInfoCollector();
    collector.collect("foo.py", fileInput);
    Assertions.assertThat(collector.get("foo.py"))
      .hasSize(1)
      .containsKey(1)
      .containsValue(new NoSonarLineInfo(1, Set.of()));

    Assertions.assertThat(collector.getLinesWithEmptyNoSonar("foo.py"))
      .hasSize(1)
      .contains(1);
  }

  @Test
  void empty_parameter_nosonar_test() {
    var astNode = PythonParser.create().parse("""
      a = 1 # NOSONAR(something,)
      """);

    var fileInput = new PythonTreeMaker().fileInput(astNode);

    var collector = new NoSonarLineInfoCollector();
    collector.collect("foo.py", fileInput);
    Assertions.assertThat(collector.get("foo.py"))
      .hasSize(1)
      .containsKey(1)
      .containsValue(new NoSonarLineInfo(1, Set.of("something")));
  }

  @Test
  void no_parameters_nosonar_test() {
    var astNode = PythonParser.create().parse("""
      a = 1 # NOSONAR
      """);

    var fileInput = new PythonTreeMaker().fileInput(astNode);

    var collector = new NoSonarLineInfoCollector();
    collector.collect("foo.py", fileInput);
    Assertions.assertThat(collector.get("foo.py"))
      .hasSize(1)
      .containsKey(1)
      .containsValue(new NoSonarLineInfo(1, Set.of()));

    Assertions.assertThat(collector.getLinesWithEmptyNoSonar("foo.py"))
      .hasSize(1)
      .contains(1);
  }

  @Test
  void multiline_nosonar_test() {
    var astNode = PythonParser.create().parse("""
      ""\"
      1
      2
      3
      ""\" # NOSONAR
      """);

    var fileInput = new PythonTreeMaker().fileInput(astNode);

    var collector = new NoSonarLineInfoCollector();
    collector.collect("foo.py", fileInput);
    Assertions.assertThat(collector.get("foo.py"))
      .hasSize(5)
      .containsOnly(
        Map.entry(1, new NoSonarLineInfo(1, Set.of())),
        Map.entry(2, new NoSonarLineInfo(2, Set.of())),
        Map.entry(3, new NoSonarLineInfo(3, Set.of())),
        Map.entry(4, new NoSonarLineInfo(4, Set.of())),
        Map.entry(5, new NoSonarLineInfo(5, Set.of()))
      );

    Assertions.assertThat(collector.getLinesWithEmptyNoSonar("foo.py"))
      .hasSize(5)
      .contains(1, 2, 3, 4, 5);
  }

}
