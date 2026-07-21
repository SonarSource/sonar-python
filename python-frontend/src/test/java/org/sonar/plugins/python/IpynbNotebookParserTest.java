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

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.event.Level;
import org.sonar.api.batch.fs.InputFile;
import com.sonarsource.scanner.engine.sensor.test.fixtures.TestInputFileBuilder;
import org.sonar.api.internal.apachecommons.lang3.StringUtils;
import org.sonar.api.testfixtures.log.LogTesterJUnit5;
import org.sonar.python.IPythonLocation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.sonar.plugins.python.NotebookTestUtils.mapToColumnMappingList;

class IpynbNotebookParserTest {

  @RegisterExtension
  public LogTesterJUnit5 logTester = new LogTesterJUnit5().setLevel(Level.DEBUG);

  private final File baseDir = new File("src/test/resources/org/sonar/plugins/python").getAbsoluteFile();

  public static PythonInputFile createInputFile(File baseDir, String name, InputFile.Status status, InputFile.Type type) {
    try {
      return new PythonInputFileImpl(TestInputFileBuilder.create("moduleKey", name)
        .setModuleBaseDir(baseDir.toPath())
        .setCharset(StandardCharsets.UTF_8)
        .setStatus(status)
        .setType(type)
        .setLanguage("py")
        .initMetadata(Files.readString(new File(baseDir, name).toPath()))
        .build());
    } catch (IOException e) {
      throw new IllegalStateException("Cannot read " + name + " from base directory" + baseDir, e);
    }
  }
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

  @ParameterizedTest
  @ValueSource(strings = {
    // "source" is a plain empty string ("", not an array). Used to leave the generated blank line
    // without a locationMap entry, drifting the cell delimiter's line onto the wrong physical line and
    // later crashing with "No IPythonLocation found for line N" once the tree maker enriched it.
    "notebook_empty_string_source.ipynb",
    // "source" is an empty array ([]).
    "notebook_empty_array_source.ipynb",
    // "source" is an array containing a single empty string ([""]).
    "notebook_array_with_empty_string_source.ipynb"
  })
  void testParseNotebookWithEmptySourceCell(String notebookFile) throws IOException {
    var inputFile = createInputFile(baseDir, notebookFile, InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);
    assertThat(resultOptional).isPresent();
    var result = resultOptional.get();

    int lineCount = result.contents().split("\n", -1).length;
    assertThat(result.locationMap().keySet()).containsExactlyInAnyOrderElementsOf(
      java.util.stream.IntStream.rangeClosed(1, lineCount).boxed().toList());

    var ast = org.sonar.python.parser.PythonParser.createIPythonParser().parse(result.contents());
    assertThatCode(() -> new org.sonar.python.tree.IPythonTreeMaker(result.locationMap()).fileInput(ast))
      .doesNotThrowAnyException();
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
    assertThat(logTester.logs(Level.DEBUG)).contains("Skipping notebook 'notebook_mojo.ipynb': unsupported language 'mojo'");
  }

  @Test
  void testParseNotebookWithCellLanguage() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_with_cell_language.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    // Should be parsed as Python despite "sparksql" language in cell metadata
    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();
    // Should contain content from all code cells (including the SQL magic cell which is still a code cell)
    assertThat(result.contents()).contains("x = 1");
    assertThat(result.contents()).contains("y = 2");
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
  void testParseNotebookWithMultilineArrayElementWithoutTrailingNewline() throws IOException {
    // "source": ["x = 1\ny", " = 2"] - the first element has no trailing newline, so its last split
    // line ("y") is continued, not terminated, by the next array element (" = 2"), giving 2 python lines
    // ("x = 1" and "y = 2") rather than 3. Before the fix, each array element always got its own
    // locationMap entry regardless of whether the previous one ended with a newline, producing one more
    // entry than there are physical lines and misaligning every location after it.
    var inputFile = createInputFile(baseDir, "notebook_multiline_string_in_array_no_trailing_newline.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();
    assertThat(result.contents()).hasLineCount(3);
    assertThat(result.locationMap().keySet()).hasSize(3);
    //"x = 1"
    assertThat(result.locationMap()).extractingByKey(1).isEqualTo(new IPythonLocation(9, 9, List.of(), true));
    //"y = 2" ("y" starts the line, " = 2" from the next array element continues it)
    assertThat(result.locationMap()).extractingByKey(2).isEqualTo(new IPythonLocation(9, 16, List.of(), true));
    // the cell delimiter
    assertThat(result.locationMap()).extractingByKey(3).isEqualTo(new IPythonLocation(10, 9, List.of(), false));
  }

  @Test
  void testParseNotebookWithMultilineStringInSourceArray() throws IOException {
    var inputFile = createInputFile(baseDir, "notebook_multiline_string_in_array.ipynb", InputFile.Status.CHANGED, InputFile.Type.MAIN);

    var resultOptional = IpynbNotebookParser.parseNotebook(inputFile);

    assertThat(resultOptional).isPresent();

    var result = resultOptional.get();
    // 11 python lines packed into the single "source" array element, plus the cell delimiter.
    // Before the fix, only 2 entries were produced (the whole array element counted as a single line),
    // which made TokenEnricher fail with "No IPythonLocation found for line 3".
    assertThat(result.contents()).hasLineCount(12);
    assertThat(result.locationMap().keySet()).hasSize(12);
    //"print(\"Test 1\")"
    assertThat(result.locationMap()).extractingByKey(1).isEqualTo(new IPythonLocation(9, 9, mapToColumnMappingList(Map.of(6, 1, 13, 1)), true));
    //"testA = \"1\""
    assertThat(result.locationMap()).extractingByKey(2).isEqualTo(new IPythonLocation(9, 28, mapToColumnMappingList(Map.of(8, 1, 10, 1)), true));
    // two consecutive empty lines
    assertThat(result.locationMap()).extractingByKey(7).isEqualTo(new IPythonLocation(9, 111, List.of(), true));
    assertThat(result.locationMap()).extractingByKey(8).isEqualTo(new IPythonLocation(9, 113, List.of(), true));
    //"print(testAll)" (last content line, no trailing newline in the JSON value)
    assertThat(result.locationMap()).extractingByKey(11).isEqualTo(new IPythonLocation(9, 150, List.of(), true));
    // the cell delimiter
    assertThat(result.locationMap()).extractingByKey(12).isEqualTo(new IPythonLocation(9, 164, List.of(), false));
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
