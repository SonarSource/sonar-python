/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.plugins.python;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.internal.apachecommons.lang3.StringUtils;
import org.sonar.python.IPythonLocation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.plugins.python.TestUtils.createInputFile;
import static org.sonar.plugins.python.TestUtils.mapToColumnMappingList;

class IpynbNotebookParserTest {
  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python").getAbsoluteFile();

  @Test
  void testParseNotebook() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();

    assertThat(result.locationMap().keySet()).hasSize(27);
    assertThat(result.contents()).hasLineCount(27);
    assertThat(StringUtils.countMatches(result.contents(), IpynbNotebookParser.SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER))
      .isEqualTo(7);
    assertThat(result.locationMap()).extracting(map -> map.get(1)).isEqualTo(new IPythonLocation(17, 5));
    //"    print \"not none\"\n"
    assertThat(result.locationMap()).extracting(map -> map.get(3)).isEqualTo(new IPythonLocation(19, 5,
      mapToColumnMappingList(Map.of(10, 1, 19, 1))));
    //"source": "#Some code\nprint(\"hello world\\n\")",
    assertThat(result.locationMap()).extracting(map -> map.get(16)).isEqualTo(new IPythonLocation(64, 14, List.of(), true));
    assertThat(result.locationMap()).extracting(map -> map.get(17)).isEqualTo(new IPythonLocation(64, 26, mapToColumnMappingList(Map.of(6
      , 1, 18, 1, 20, 1)), true));
    //"source": "print(\"My\\ntext\")\nprint(\"Something else\\n\")"
    assertThat(result.locationMap()).extracting(map -> map.get(22)).isEqualTo(new IPythonLocation(83, 14, mapToColumnMappingList(Map.of(6
      , 1, 9, 1, 15, 1)), true));
    assertThat(result.locationMap()).extracting(map -> map.get(23)).isEqualTo(new IPythonLocation(83, 36, mapToColumnMappingList(Map.of(6
      , 1, 21, 1, 23, 1)), true));

    //"source": "a = \"A bunch of characters \\n \\f \\r \\  \"\nb = None"
    assertThat(result.locationMap()).extracting(map -> map.get(25))
      .isEqualTo(new IPythonLocation(90, 14, mapToColumnMappingList(Map.of(4, 1, 27, 1, 30, 1, 33, 1, 36, 1, 39, 1)), true));
    assertThat(result.locationMap()).extracting(map -> map.get(26)).isEqualTo(new IPythonLocation(90, 62, List.of(), true));
    // last line with the cell delimiter which contains the EOF token
    assertThat(result.locationMap()).extracting(map -> map.get(27)).isEqualTo(new IPythonLocation(90, 14, List.of()));
  }

  @Test
  void testParseNotebookWithEscapedChars() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_with_escaped_chars.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();

    assertThat(result.locationMap().keySet()).hasSize(2);
    assertThat(result.contents()).hasLineCount(2);
    assertThat(StringUtils.countMatches(result.contents(), IpynbNotebookParser.SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER))
      .isEqualTo(1);

    //"source":  "\t \b \f \"\""
    assertThat(result.locationMap()).extracting(map -> map.get(1))
      .isEqualTo(new IPythonLocation(14, 15, mapToColumnMappingList(
          Map.ofEntries(
            Map.entry(0, 1),
            Map.entry(2, 1),
            Map.entry(4, 1),
            Map.entry(6, 1),
            Map.entry(7, 1)
          )
        ), true));

  }

  @Test
  void testParseNotebookWithEmptyLines() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_with_empty_lines.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();

    assertThat(result.locationMap().keySet()).hasSize(4);
    assertThat(result.contents()).hasLineCount(4);
    assertThat(StringUtils.countMatches(result.contents(), IpynbNotebookParser.SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER))
      .isEqualTo(1);
    assertThat(result.locationMap()).extracting(map -> map.get(3)).isEqualTo(new IPythonLocation(11, 5));

    // last line with the cell delimiter which contains the EOF token
    assertThat(result.locationMap()).extracting(map -> map.get(4)).isEqualTo(new IPythonLocation(11, 5));
  }

  @Test
  void testParseInvalidNotebook() {
    var inputFile = createInputFile(baseDir, "invalid_notebook.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    assertThatThrownBy(() -> IpynbNotebookParser.parseNotebook(inputFile))
      .isInstanceOf(IllegalStateException.class)
      .hasMessageContaining("Unexpected token");
  }

  @Test
  void testParseMojoNotebook() {
    var inputFile = createInputFile(baseDir, "notebook_mojo.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isEmpty();
  }

  @Test
  void testParseNotebookWithNoLanguage() {
    var inputFile = createInputFile(baseDir, "notebook_no_language.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();
  }

  @Test
  void testParseNotebookWithExtraLineEndInArray() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_extra_line.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();
    assertThat(result.locationMap()).hasSize(3);
    assertThat(result.contents()).hasLineCount(3);
  }

  @Test
  void testParseNotebookSingleLine() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_single_line.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();
    assertThat(result.locationMap()).hasSize(9);
    assertThat(result.contents()).hasLineCount(9);
    // position of variable t
    assertThat(result.locationMap().get(4).column()).isEqualTo(451);

    // First and second line
    assertThat(result.locationMap()).containsEntry(1, new IPythonLocation(1, 382, List.of(), true));
    assertThat(result.locationMap()).containsEntry(2, new IPythonLocation(1, 428, List.of(), true));

    assertThat(result.locationMap()).containsEntry(6, new IPythonLocation(1, 559, mapToColumnMappingList(Map.of(0, 1, 1, 1, 2, 1)), true));
    assertThat(result.locationMap()).containsEntry(7, new IPythonLocation(1, 609, List.of(), true));
    assertThat(result.locationMap()).containsEntry(8, new IPythonLocation(1, 636, mapToColumnMappingList(Map.of(1, 1, 2, 1, 0, 1)), true));
  }

  @Test
  void testParseNotebook1() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_no_code.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();
    assertThat(result.locationMap()).isEmpty();
    assertThat(result.contents()).isEmpty();
  }
}
