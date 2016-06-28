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
    
    file = new File(dir + "/highlighting.py");
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
  }
  
  @Test
  public void comment() throws Exception {
    checkOnRange(6, 0, 19, TypeOfText.COMMENT);
    
    checkOnRange(9, 10, 15, TypeOfText.COMMENT);
  }
  
  @Test
  public void docStringTripleSimpleQuotes() throws Exception {
    // triple simple quotes
    checkOnRange(14, 0, 15, TypeOfText.STRING);
    
    // triple double quotes
    checkOnRange(16, 0, 15, TypeOfText.STRING);
  }
  
  /**
   * Checks the highlighting on a range of columns.
   * The range is the columns of the token. 
   */
  private void checkOnRange(int line, int firstColumn, int length, TypeOfText expectedTypeOfText) {
    String componentKey = "moduleKey:" + file.getName();
    
    // check that every column of the token is highlighted (and with the expected type)
    for (int column = firstColumn; column < firstColumn + length; column++) {
      List<TypeOfText> foundTypeOfTexts = context.highlightingTypeAt(componentKey, line, column);
      String name = "number of TypeOfTexts at line " + line + " and column " + column;
      assertThat(foundTypeOfTexts).as(name).hasSize(1);
      assertThat(foundTypeOfTexts.get(0)).isEqualTo(expectedTypeOfText);
    }
    
    // check that the column before the token is not highlighted
    if (firstColumn != 1) {
      int column = firstColumn - 1;
      List<TypeOfText> foundTypeOfTexts = context.highlightingTypeAt(componentKey, line, column);
      String name = "number of TypeOfTexts at line " + line + " and column " + column + " (= before the token)";
      assertThat(foundTypeOfTexts).as(name).hasSize(0);
    }
    
    // check that the column after the token is not highlighted
    int column = firstColumn + length;
    List<TypeOfText> foundTypeOfTexts = context.highlightingTypeAt(componentKey, line, column);
    String name = "number of TypeOfTexts at line " + line + " and column " + column + " (= after the token)";
    assertThat(foundTypeOfTexts).as(name).hasSize(0);
  }
  
}
