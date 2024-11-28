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
    assertThat(result.locationMap()).extractingByKey(1).isEqualTo(new IPythonLocation(17, 5));
    //"    print \"not none\"\n"
    assertThat(result.locationMap()).extractingByKey(3).isEqualTo(new IPythonLocation(19, 5,
      mapToColumnMappingList(Map.of(10, 1, 19, 1))));
    //"source": "#Some code\nprint(\"hello world\\n\")",
    assertThat(result.locationMap()).extractingByKey(16).isEqualTo(new IPythonLocation(64, 14, List.of(), true));
    assertThat(result.locationMap()).extractingByKey(17).isEqualTo(new IPythonLocation(64, 26, mapToColumnMappingList(Map.of(6
      , 1, 18, 1, 20, 1)), true));
    //"source": "print(\"My\\ntext\")\nprint(\"Something else\\n\")"
    assertThat(result.locationMap()).extractingByKey(22).isEqualTo(new IPythonLocation(83, 14, mapToColumnMappingList(Map.of(6
      , 1, 9, 1, 15, 1)), true));
    assertThat(result.locationMap()).extractingByKey(23).isEqualTo(new IPythonLocation(83, 36, mapToColumnMappingList(Map.of(6
      , 1, 21, 1, 23, 1)), true));

    //"source": "a = \"A bunch of characters \\n \\f \\r \\  \"\nb = None"
    assertThat(result.locationMap()).extractingByKey(25)
      .isEqualTo(new IPythonLocation(90, 14, mapToColumnMappingList(Map.of(4, 1, 27, 1, 30, 1, 33, 1, 36, 1, 39, 1)), true));
    assertThat(result.locationMap()).extractingByKey(26).isEqualTo(new IPythonLocation(90, 62, List.of(), true));
    // last line with the cell delimiter which contains the EOF token the column of this token should be at the end of the previous line 
    assertThat(result.locationMap()).extractingByKey(27).isEqualTo(new IPythonLocation(90, 72, List.of(), false));
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

    assertThat(result.locationMap().keySet()).hasSize(5);
    assertThat(result.contents()).hasLineCount(5);
    assertThat(StringUtils.countMatches(result.contents(), IpynbNotebookParser.SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER))
      .isEqualTo(1);
    assertThat(result.locationMap()).extractingByKey(4).extracting(IPythonLocation::line).isEqualTo(11);
    assertThat(result.locationMap()).extractingByKey(4).extracting(IPythonLocation::column).isEqualTo(5);

    // last line with the cell delimiter which contains the EOF token
    assertThat(result.locationMap()).extractingByKey(5).isEqualTo(new IPythonLocation(11, 5));
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
  void testDifferentJsonRepresentationOfEmptyLine() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_extra_line.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    var inputFileExtraLineExplicit = createInputFile(baseDir, "notebook_extra_line_explicit.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    var inputFileExtraLineExplicitSingleLine = createInputFile(baseDir, "notebook_extra_line_compressed.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var notebook = IpynbNotebookParser.parseNotebook(inputFile);
    var notebookExtraLineExplicit = IpynbNotebookParser.parseNotebook(inputFileExtraLineExplicit);
    var notebookExtraLineExplicitSingleLine = IpynbNotebookParser.parseNotebook(inputFileExtraLineExplicitSingleLine);

    assertThat(notebook).isPresent();
    assertThat(notebookExtraLineExplicit).isPresent();
    assertThat(notebookExtraLineExplicitSingleLine).isPresent();

    assertThat(notebook.get().contents()).isEqualTo(notebookExtraLineExplicit.get().contents());
    assertThat(notebook.get().contents()).isEqualTo(notebookExtraLineExplicitSingleLine.get().contents());
  }

  @Test
  void testParseNotebookEndingWithEmptyLine() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_extra_line.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();
    assertThat(result.locationMap()).hasSize(4);
    assertThat(result.contents()).hasLineCount(4);
    // The empty line
    assertThat(result.locationMap()).extractingByKey(3).extracting(IPythonLocation::line).isEqualTo(19);
    assertThat(result.locationMap()).extractingByKey(3).extracting(IPythonLocation::column).isEqualTo(5);
    // The delimiter
    assertThat(result.locationMap()).extractingByKey(4).extracting(IPythonLocation::line).isEqualTo(19);
    assertThat(result.locationMap()).extractingByKey(4).extracting(IPythonLocation::column).isEqualTo(5);
      
    inputFile = createInputFile(baseDir, "notebook_extra_line_compressed.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);
    resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    result = resultOptional.get();
    assertThat(result.locationMap()).hasSize(4);
    assertThat(result.contents()).hasLineCount(4);
    // The empty line
    assertThat(result.locationMap()).extractingByKey(3).extracting(IPythonLocation::line).isEqualTo(1);
    assertThat(result.locationMap()).extractingByKey(3).extracting(IPythonLocation::column).isEqualTo(317);
    // The delimiter is added after the empty line
    assertThat(result.locationMap()).extractingByKey(4).extracting(IPythonLocation::line).isEqualTo(1);
    assertThat(result.locationMap()).extractingByKey(4).extracting(IPythonLocation::column).isEqualTo(319);
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
