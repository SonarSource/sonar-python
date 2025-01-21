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
package org.sonar.plugins.python;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.highlighting.TypeOfText;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.python.IPythonLocation;
import org.sonar.python.TestPythonVisitorRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.plugins.python.NotebookTestUtils.mapToColumnMappingList;

class PythonHighlighterTest {

  private SensorContextTester context;

  private File file;
  private File notebookFile;
  private File notebookFileSingleLine;
  private DefaultInputFile notebookInputFile;
  private DefaultInputFile notebookInputFileSingleLine;
  private String dir = "src/test/resources/org/sonar/plugins/python";

  @BeforeEach
  void scanFile() {

    file = new File(dir, "/pythonHighlighter.py");
    DefaultInputFile inputFile = TestInputFileBuilder.create("moduleKey", file.getName())
      .initMetadata(TestUtils.fileContent(file, StandardCharsets.UTF_8))
      .build();

    notebookFile = new File(dir, "/notebookHighlighter.ipynb");
    notebookInputFile = TestInputFileBuilder.create("moduleKey", notebookFile.getName())
      .initMetadata(TestUtils.fileContent(notebookFile, StandardCharsets.UTF_8))
      .build();

    notebookFileSingleLine = new File(dir, "/notebookHighlighterSingleLine.ipynb");
    notebookInputFileSingleLine = TestInputFileBuilder.create("moduleKey", notebookFileSingleLine.getName())
      .initMetadata(TestUtils.fileContent(notebookFileSingleLine, StandardCharsets.UTF_8))
      .build();

    context = SensorContextTester.create(new File(dir));
    context.fileSystem().add(inputFile);
    context.fileSystem().add(notebookInputFile);
    context.fileSystem().add(notebookInputFileSingleLine);

    PythonHighlighter pythonHighlighter = new PythonHighlighter(context, new PythonInputFileImpl(inputFile));
    TestPythonVisitorRunner.scanFile(file, pythonHighlighter);
  }

  @Test
  void keyword() {
    // def
    checkOnRange(8, 0, 3, file, TypeOfText.KEYWORD);

    // if
    checkOnRange(12, 0, 2, file, TypeOfText.KEYWORD);

    // or
    checkOnRange(12, 12, 2, file, TypeOfText.KEYWORD);

    // or
    checkOnRange(12, 24, 2, file, TypeOfText.KEYWORD);

    // continue
    checkOnRange(12, 37, 8, file, TypeOfText.KEYWORD);

    // pass
    checkOnRange(9, 4, 4, file, TypeOfText.KEYWORD);

    // async
    checkOnRange(95, 0, 5, file, TypeOfText.KEYWORD);

    // await
    checkOnRange(98, 0, 5, file, TypeOfText.KEYWORD);

    // match
    checkOnRange(100, 0, 5, file, TypeOfText.KEYWORD);

    // case
    checkOnRange(101, 4, 4, file, TypeOfText.KEYWORD);
  }

  @Test
  void stringLiteral() {
    // "some string"
    checkOnRange(4, 4, 13, file, TypeOfText.STRING);

    // 'some string'
    checkOnRange(18, 4, 13, file, TypeOfText.STRING);

    // triple simple quotes
    checkOnRange(14, 0, 15, file, TypeOfText.STRING);

    // triple double quotes
    checkOnRange(16, 0, 15, file, TypeOfText.STRING);

    // y = """ some string
    // that extends
    // on several
    // lines
    // """
    check(20, 3, file, null);
    check(20, 4, file, TypeOfText.STRING);
    check(21, 10, file, TypeOfText.STRING);
    check(22, 10, file, TypeOfText.STRING);
    check(23, 10, file, TypeOfText.STRING);
    check(24, 6, file, TypeOfText.STRING);
    check(24, 7, file, null);

    // values=["""long...
    // ...string 1""", 3.14, "short string 2"]
    check(26, 7, file, null);
    check(26, 8, file, TypeOfText.STRING);
    check(27, 17, file, TypeOfText.STRING);
    check(27, 18, file, null);
    checkOnRange(27, 26, 16, file, TypeOfText.STRING);
  }

  @Test
  void docStrings() {
    // docstrings and non-docstrings
    check(1, 0, file, TypeOfText.STRUCTURED_COMMENT);
    checkOnRange(2, 0, 22, file, TypeOfText.STRUCTURED_COMMENT);
    checkOnRange(50, 4, 28, file, TypeOfText.STRUCTURED_COMMENT);
    check(54, 4, file, TypeOfText.STRUCTURED_COMMENT);
    check(55, 4, file, TypeOfText.STRUCTURED_COMMENT);
    check(56, 4, file, TypeOfText.STRUCTURED_COMMENT);
    checkOnRange(57, 0, 7, file, TypeOfText.STRUCTURED_COMMENT);
    checkOnRange(58, 4, 64, file, TypeOfText.STRING);
    checkOnRange(60, 4, 23, file, TypeOfText.STRING);
    checkOnRange(64, 4, 31, file, TypeOfText.STRUCTURED_COMMENT);
    checkOnRange(67, 4, 69, file, TypeOfText.STRING);
    check(70, 14, file, TypeOfText.STRUCTURED_COMMENT);
    check(71, 14, file, TypeOfText.STRUCTURED_COMMENT);
    checkOnRange(72, 0, 25, file, TypeOfText.STRUCTURED_COMMENT);
    checkOnRange(77, 8, 23, file, TypeOfText.STRING);
    checkOnRange(79, 12, 23, file, TypeOfText.STRING);
    checkOnRange(84, 0, 23, file, TypeOfText.STRING);
    checkOnRange(87, 4, 23, file, TypeOfText.STRING);

    checkOnRange(93, 11, 17, file, TypeOfText.STRUCTURED_COMMENT);
    checkOnRange(91, 8, 17, file, TypeOfText.STRING);
  }

  @Test
  void comment() {
    checkOnRange(6, 0, 19, file, TypeOfText.COMMENT);
    checkOnRange(9, 10, 15, file, TypeOfText.COMMENT);
  }

  @Test
  void number() {
    // 34
    checkOnRange(29, 0, 2, file, TypeOfText.CONSTANT);

    // -35 (negative numbers are parsed as 2 tokens)
    checkOnRange(31, 1, 2, file, TypeOfText.CONSTANT);

    // 20000000000000L
    checkOnRange(33, 0, 15, file, TypeOfText.CONSTANT);

    // 1000l
    checkOnRange(35, 0, 5, file, TypeOfText.CONSTANT);

    // 89e4
    checkOnRange(37, 0, 4, file, TypeOfText.CONSTANT);

    // y = -45.4 + 67e8 - 78.562E-09
    checkOnRange(39, 4, 4, file, TypeOfText.CONSTANT);
    checkOnRange(39, 11, 4, file, TypeOfText.CONSTANT);
    checkOnRange(39, 18, 10, file, TypeOfText.CONSTANT);

    // 4.55j
    checkOnRange(41, 0, 5, file, TypeOfText.CONSTANT);

    // -4.55j
    checkOnRange(43, 1, 5, file, TypeOfText.CONSTANT);

    // 3J
    checkOnRange(45, 0, 2, file, TypeOfText.CONSTANT);

    // 23.3e-7J
    checkOnRange(47, 0, 8, file, TypeOfText.CONSTANT);
  }

  @Test
  void highlightingNotebooks() {
    String pythonContent = """
      def foo():
          pass
      a = "test" # comment
      # test \\n \\\\n test
      b = 3J
      c = \t4.2
      #SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER""";
    var locations = Map.of(
      1, new IPythonLocation(9, 5),
      2, new IPythonLocation(10, 5),
      3, new IPythonLocation(11, 5, mapToColumnMappingList(Map.of(-1, 2, 4, 1, 9, 1))),
      4, new IPythonLocation(12, 5, mapToColumnMappingList(Map.of(-1, 3, 7, 1, 10, 1, 11, 1))),
      5, new IPythonLocation(13, 5),
      6, new IPythonLocation(14, 5, mapToColumnMappingList(Map.of(4, 1))),
      7, new IPythonLocation(14, 5)); //EOF Token
    PythonHighlighter pythonHighlighter = new PythonHighlighter(context, new GeneratedIPythonFile(notebookInputFile, pythonContent,
      locations));
    TestPythonVisitorRunner.scanNotebookFile(notebookFile, locations, pythonContent, pythonHighlighter);
    // def
    checkOnRange(9, 5, 3, notebookFile, TypeOfText.KEYWORD);
    // pass
    checkOnRange(10, 9, 4, notebookFile, TypeOfText.KEYWORD);
    // \"test\"
    checkOnRange(11, 9, 8, notebookFile, TypeOfText.STRING);
    // # comment
    checkOnRange(11, 18, 9, notebookFile, TypeOfText.COMMENT);
    // # test \\n \\\\n test
    checkOnRange(12, 5, 21, notebookFile, TypeOfText.COMMENT);
    // 3J
    checkOnRange(13, 9, 2, notebookFile, TypeOfText.CONSTANT);
    // 4.2
    checkOnRange(14, 11, 3, notebookFile, TypeOfText.CONSTANT);
  }

  @Test
  void highlightingNotebooksSingleLine() {
    String pythonContent = "def foo():\n    pass\na = 2 # comment\n#SONAR_PYTHON_NOTEBOOK_CELL_DELIMITER";
    var locations = Map.of(
      1, new IPythonLocation(1, 93, List.of(), true),
      2, new IPythonLocation(1, 108, List.of(), true),
      3, new IPythonLocation(1, 121, List.of(), true),
      4, new IPythonLocation(1, 93, List.of(), true)); //EOF Token
    PythonHighlighter pythonHighlighter = new PythonHighlighter(context, new GeneratedIPythonFile(notebookInputFileSingleLine,
      pythonContent, locations));
    TestPythonVisitorRunner.scanNotebookFile(notebookFileSingleLine, locations, pythonContent, pythonHighlighter);
    // def
    checkOnRange(1, 93, 3, notebookFileSingleLine, TypeOfText.KEYWORD);
    // pass
    checkOnRange(1, 112, 4, notebookFileSingleLine, TypeOfText.KEYWORD);
    // 2
    checkOnRange(1, 125, 1, notebookFileSingleLine, TypeOfText.CONSTANT);
    // # comment
    checkOnRange(1, 127, 9, notebookFileSingleLine, TypeOfText.COMMENT);
  }

  /**
   * Checks the highlighting of a range of columns. The first column of a line has index 0.
   * The range is the columns of the token.
   */
  private void checkOnRange(int line, int firstColumn, int length, File file, TypeOfText expectedTypeOfText) {
    // check that every column of the token is highlighted (and with the expected type)
    for (int column = firstColumn; column < firstColumn + length; column++) {
      checkInternal(line, column, "", file, expectedTypeOfText);
    }

    // check that the column before the token is not highlighted
    if (firstColumn != 0) {
      checkInternal(line, firstColumn - 1, " (= before the token)", file, null);
    }

    // check that the column after the token is not highlighted
    checkInternal(line, firstColumn + length, " (= after the token)", file, null);
  }

  /**
   * Checks the highlighting of one column. The first column of a line has index 0.
   */
  private void check(int line, int column, File file, @Nullable TypeOfText expectedTypeOfText) {
    checkInternal(line, column, "", file, expectedTypeOfText);
  }

  private void checkInternal(int line, int column, String messageComplement, File file, @Nullable TypeOfText expectedTypeOfText) {
    String componentKey = "moduleKey:" + file.getName();
    List<TypeOfText> foundTypeOfTexts = context.highlightingTypeAt(componentKey, line, column);

    int expectedNumberOfTypeOfText = expectedTypeOfText == null ? 0 : 1;
    String message = "number of TypeOfTexts at line " + line + " and column " + column + messageComplement;
    assertThat(foundTypeOfTexts).as(message).hasSize(expectedNumberOfTypeOfText);
    if (expectedNumberOfTypeOfText > 0) {
      message = "found TypeOfTexts at line " + line + " and column " + column + messageComplement;
      assertThat(foundTypeOfTexts.get(0)).as(message).isEqualTo(expectedTypeOfText);
    }
  }

}
