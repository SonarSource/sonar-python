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
package org.sonar.plugins.python;

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.NotebookDialect.DATABRICKS;
import static org.sonar.plugins.python.NotebookDialect.IPYTHON;

class NotebookMagicClassifierTest {

  @ParameterizedTest(name = "{0}")
  @MethodSource({"ordinarySources", "nonDirectivePositions"})
  void keep_sources_without_a_leading_magic_visible(String source) {
    assertThat(NotebookMagicClassifier.isOpaqueCell(DATABRICKS, source)).isFalse();
    assertThat(NotebookMagicClassifier.isOpaqueCell(IPYTHON, source)).isFalse();
  }

  static Stream<String> ordinarySources() {
    return Stream.of(
      "",
      "answer = 42",
      "# %sql\nanswer = 42",
      "print('%sql')");
  }

  @ParameterizedTest(name = "{0}: {1}")
  @MethodSource("dialectsAndCellMagicForms")
  void treat_double_percent_magic_as_opaque(NotebookDialect dialect, String source) {
    assertThat(NotebookMagicClassifier.isOpaqueCell(dialect, source)).isTrue();
  }

  static Stream<Arguments> dialectsAndCellMagicForms() {
    return Stream.of(IPYTHON, DATABRICKS)
      .flatMap(dialect -> Stream.of(
        Arguments.of(dialect, "%%anything inline body"),
        Arguments.of(dialect, "%%anything\nbody")));
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("databricksMagicCases")
  void identify_opaque_databricks_cells(String description, String source, boolean expected) {
    assertThat(NotebookMagicClassifier.isOpaqueCell(DATABRICKS, source)).isEqualTo(expected);
  }

  static Stream<Arguments> databricksMagicCases() {
    return Stream.of(
      Arguments.of("inline SQL cell", "%sql SELECT 1\nFROM values", true),
      Arguments.of("multiline SQL cell", "%sql\nSELECT 1\nFROM values", true),
      Arguments.of("inline notebook run", "%run ../core/utils", true),
      Arguments.of("multiline notebook run", "%run\n../core/utils", true),
      Arguments.of("Scala language cell", "%scala\nprintln(42)", true),
      Arguments.of("non-Python language cell", "%r\nprint('hello')", true),
      Arguments.of("Markdown cell", "%md\n# Heading", true),
      Arguments.of("sandboxed Markdown cell", "%md-sandbox\n<div>not Python</div>", true),
      Arguments.of("shell cell", "%sh\necho hello", true),
      Arguments.of("filesystem command cell", "%fs\nls dbfs:/datasets", true),
      Arguments.of("skipped cell", "%skip\ninvalid Python", true),
      Arguments.of("Python language directive", "%python\nanswer = 42", false),
      Arguments.of("auxiliary line magic", "%pip install pandas", false),
      Arguments.of("unknown line magic", "%time answer = 42", false));
  }

  @ParameterizedTest(name = "%{0}")
  @MethodSource("allSinglePercentCommands")
  void keep_single_percent_magic_bodies_visible_in_ipython(String command) {
    assertThat(NotebookMagicClassifier.isOpaqueCell(IPYTHON, "%" + command + " inline body")).isFalse();
    assertThat(NotebookMagicClassifier.isOpaqueCell(IPYTHON, "%" + command + "\nbody")).isFalse();
  }

  static Stream<String> allSinglePercentCommands() {
    return Stream.of("python", "sql", "md-sandbox", "fs", "run", "pip", "foo");
  }

  @Test
  void keep_unknown_databricks_magic_bodies_visible() {
    assertThat(NotebookMagicClassifier.isOpaqueCell(DATABRICKS, "%foo inline body")).isFalse();
    assertThat(NotebookMagicClassifier.isOpaqueCell(DATABRICKS, "%foo\nbody")).isFalse();
  }

  static Stream<String> nonDirectivePositions() {
    return Stream.of(
      " %sql\nSELECT 1",
      "\t%sql\nSELECT 1",
      "\n%sql\nSELECT 1",
      "answer = 42\n%sql\nSELECT 1",
      " %%anything\nbody");
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("commandBoundaryCases")
  void match_magic_commands_on_exact_boundaries_and_case(String source) {
    assertThat(NotebookMagicClassifier.isOpaqueCell(DATABRICKS, source)).isFalse();
  }

  static Stream<String> commandBoundaryCases() {
    return Stream.of(
      "%sqlalchemy SELECT 1",
      "%pythonista print('hello')",
      "%runner ../notebook",
      "%SQL SELECT 1",
      "%Python print('hello')");
  }
}
