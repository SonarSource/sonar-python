/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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

import com.google.common.base.Charsets;
import java.io.File;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.FileMetadata;
import org.sonar.api.batch.sensor.highlighting.TypeOfText;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.python.PythonAstScanner;

import static org.fest.assertions.Assertions.assertThat;

public class PythonHighlighterTest {

  private SensorContextTester context;

  private File file;

  @Before
  @SuppressWarnings("unchecked")
  public void scanFile() {
    String dir = "src/test/resources/org/sonar/plugins/python";

    file = new File(dir + "/pythonHighlighter.py");
    DefaultInputFile inputFile = new DefaultInputFile("moduleKey", file.getName())
      .initMetadata(new FileMetadata().readMetadata(file, Charsets.UTF_8));

    context = SensorContextTester.create(new File(dir));
    context.fileSystem().add(inputFile);

    PythonHighlighter pythonHighlighter = new PythonHighlighter(context);
    PythonAstScanner.scanSingleFile(file, pythonHighlighter);
  }

  @Test
  public void keyword() throws Exception {
    // def
    checkOnRange(1, 0, 3, TypeOfText.KEYWORD);

    // if
    checkOnRange(12, 0, 2, TypeOfText.KEYWORD);

    // or
    checkOnRange(12, 12, 2, TypeOfText.KEYWORD);

    // or
    checkOnRange(12, 24, 2, TypeOfText.KEYWORD);

    // continue
    checkOnRange(12, 37, 8, TypeOfText.KEYWORD);

    // pass
    checkOnRange(2, 4, 4, TypeOfText.KEYWORD);
  }

  @Test
  public void stringLiteral() throws Exception {
    // "some string"
    checkOnRange(4, 4, 13, TypeOfText.STRING);

    // 'some string'
    checkOnRange(18, 4, 13, TypeOfText.STRING);

    // triple simple quotes
    checkOnRange(14, 0, 15, TypeOfText.STRING);

    // triple double quotes
    checkOnRange(16, 0, 15, TypeOfText.STRING);

    // y = """ some string
    // that extends
    // on several
    // lines
    // """
    check(20, 3, null);
    check(20, 4, TypeOfText.STRING);
    check(21, 10, TypeOfText.STRING);
    check(22, 10, TypeOfText.STRING);
    check(23, 10, TypeOfText.STRING);
    check(24, 6, TypeOfText.STRING);
    check(24, 7, null);

    // values=["""long...
    // ...string 1""", 3.14, "short string 2"]
    check(26, 7, null);
    check(26, 8, TypeOfText.STRING);
    check(27, 17, TypeOfText.STRING);
    check(27, 18, null);
    checkOnRange(27, 26, 16, TypeOfText.STRING);
  }

  @Test
  public void comment() throws Exception {
    checkOnRange(6, 0, 19, TypeOfText.COMMENT);
    checkOnRange(9, 10, 15, TypeOfText.COMMENT);
  }

  /**
   * Checks the highlighting of a range of columns.
   * The range is the columns of the token.
   */
  private void checkOnRange(int line, int firstColumn, int length, TypeOfText expectedTypeOfText) {
    // check that every column of the token is highlighted (and with the expected type)
    for (int column = firstColumn; column < firstColumn + length; column++) {
      checkInternal(line, column, "", expectedTypeOfText);
    }

    // check that the column before the token is not highlighted
    if (firstColumn != 1) {
      checkInternal(line, firstColumn - 1, " (= before the token)", null);
    }

    // check that the column after the token is not highlighted
    checkInternal(line, firstColumn + length, " (= after the token)", null);
  }

  /**
   * Checks the highlighting of one column.
   */
  private void check(int line, int column, TypeOfText expectedTypeOfText) {
    checkInternal(line, column, "", expectedTypeOfText);
  }

  private void checkInternal(int line, int column, String messageComplement, TypeOfText expectedTypeOfText) {
    String componentKey = "moduleKey:" + file.getName();
    List<TypeOfText> foundTypeOfTexts = context.highlightingTypeAt(componentKey, line, column);

    int expectedNumberOfTypeOfText = expectedTypeOfText == null ? 0 : 1;
    String message = "number of TypeOfTexts at line " + line + " and column " + column + messageComplement;
    assertThat(foundTypeOfTexts).as(message).hasSize(expectedNumberOfTypeOfText);
    if (expectedNumberOfTypeOfText > 0) {
      assertThat(foundTypeOfTexts.get(0)).isEqualTo(expectedTypeOfText);
    }
  }

}
