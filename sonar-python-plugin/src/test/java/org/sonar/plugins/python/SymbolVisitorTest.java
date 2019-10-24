/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import org.junit.BeforeClass;
import org.junit.Test;
import org.sonar.api.batch.fs.TextRange;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.DefaultTextPointer;
import org.sonar.api.batch.fs.internal.DefaultTextRange;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.plugins.python.api.tree.FileInput;

import static org.assertj.core.api.Assertions.assertThat;

public class SymbolVisitorTest {

  private static SensorContextTester context;
  private static String componentKey;

  @BeforeClass
  public static void scanFile() {
    File file = new File("src/test/resources/org/sonar/plugins/python/sensor", "/symbolVisitor.py");
    DefaultInputFile inputFile = TestInputFileBuilder.create("moduleKey", file.getName())
      .initMetadata(TestUtils.fileContent(file, StandardCharsets.UTF_8))
      .build();

    context = SensorContextTester.create(file);
    context.fileSystem().add(inputFile);
    componentKey = inputFile.key();

    SymbolVisitor symbolVisitor = new SymbolVisitor(context.newSymbolTable().onFile(inputFile));
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(file);
    FileInput fileInput = context.rootTree();
    fileInput.accept(symbolVisitor);
  }

  @Test
  public void symbol_visitor() {
    assertThat(context.referencesForSymbolAt(componentKey, 1, 10)).isNull();
    verifyUsages(1, 0, reference(29, 14, 29, 15), reference(30, 18, 30, 19));
    verifyUsages(2, 0, reference(3, 6, 3, 7), reference(10, 4, 10, 5));
    verifyUsages(5, 4, reference(6, 4, 6, 5), reference(7, 4, 7, 5),
      reference(8, 8, 8, 9), reference(13, 9, 13, 10));
    verifyUsages(9, 4);
    verifyUsages(16, 4);
    verifyUsages(23, 13);
    verifyUsages(19, 13, reference(22, 13, 22, 14), reference(26, 13, 26, 14));
    verifyUsages(18, 17, reference(19, 8, 19, 12));
    verifyUsages(21, 18, reference(22, 8, 22, 12), reference(23, 8, 23, 12));
    verifyUsages(25, 19, reference(26, 8, 26, 12));
    verifyUsages(28, 11, reference(28, 15, 28, 16), reference(28, 17, 28, 18));
    verifyUsages(29, 9, reference(29, 1, 29, 2));
    verifyUsages(30, 11, reference(30, 1, 30, 4));
  }

  private void verifyUsages(int line, int offset, TextRange... trs) {
    Collection<TextRange> textRanges = context.referencesForSymbolAt(componentKey, line, offset);
    assertThat(textRanges).containsExactly(trs);
  }

  private static TextRange reference(int lineStart, int columnStart, int lineEnd, int columnEnd) {
    return new DefaultTextRange(new DefaultTextPointer(lineStart, columnStart), new DefaultTextPointer(lineEnd, columnEnd));
  }
}
