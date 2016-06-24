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
    file = new File("src/test/resources/org/sonar/plugins/python/highlighting.py");
    DefaultInputFile inputFile = new DefaultInputFile("moduleKey", file.getName())
      .initMetadata(new FileMetadata().readMetadata(file, Charsets.UTF_8));
    
    context = SensorContextTester.create(new File("src/test/resources/org/sonar/plugins/python").getAbsoluteFile());
    context.fileSystem().add(inputFile);
    
    PythonHighlighter pythonHighlighter = new PythonHighlighter(context);
    PythonAstScanner.scanSingleFile(file, pythonHighlighter);
  }

  @Test
  public void keyword_def() throws Exception {
    List<TypeOfText> typeOfTexts = getTypesOfText(1, 1, 1);

    assertThat(typeOfTexts.get(0)).isEqualTo(TypeOfText.KEYWORD);
  }

  @Test
  public void keyword_four() throws Exception {
    List<TypeOfText> typeOfTexts;
    
    // if
    typeOfTexts = getTypesOfText(12, 1, 1);
    assertThat(typeOfTexts.get(0)).isEqualTo(TypeOfText.KEYWORD);
    
    // or
    typeOfTexts = getTypesOfText(12, 13, 1);
    assertThat(typeOfTexts.get(0)).isEqualTo(TypeOfText.KEYWORD);
    
    // or
    typeOfTexts = getTypesOfText(12, 25, 1);
    assertThat(typeOfTexts.get(0)).isEqualTo(TypeOfText.KEYWORD);
    
    // continue
    typeOfTexts = getTypesOfText(12, 38, 1);
    assertThat(typeOfTexts.get(0)).isEqualTo(TypeOfText.KEYWORD);
  }
  
  @Test
  public void keyword_pass() throws Exception {
    List<TypeOfText> typeOfTexts = getTypesOfText(2, 4, 1);
    
    assertThat(typeOfTexts.get(0)).isEqualTo(TypeOfText.KEYWORD);
  }
  
  @Test
  public void string_literal() throws Exception {
    List<TypeOfText> typeOfTexts = getTypesOfText(4, 5, 1);
    
    assertThat(typeOfTexts.get(0)).isEqualTo(TypeOfText.STRING);
  }
  
  @Test
  public void comment() throws Exception {
    List<TypeOfText> typeOfTexts = getTypesOfText(6, 6, 1);
    
    assertThat(typeOfTexts.get(0)).isEqualTo(TypeOfText.COMMENT);
  }
  
  @Test
  public void comment_misplaced() throws Exception {
    List<TypeOfText> typeOfTexts = getTypesOfText(9, 18, 1);
    
    assertThat(typeOfTexts.get(0)).isEqualTo(TypeOfText.COMMENT);
  }
  
  private List<TypeOfText>  getTypesOfText(int line, int column, int expectedSize) {
    List<TypeOfText> typeOfTexts = context.highlightingTypeAt("moduleKey:" + file.getName(), line, column);

    assertThat(typeOfTexts).as("types of text").hasSize(1);
    
    return typeOfTexts;
  }
  
}
