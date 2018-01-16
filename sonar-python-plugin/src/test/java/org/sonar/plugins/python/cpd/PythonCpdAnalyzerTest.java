/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.plugins.python.cpd;

import com.google.common.base.Charsets;
import java.io.File;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.duplications.internal.pmd.TokensLine;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.TestUtils;
import org.sonar.python.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;

public class PythonCpdAnalyzerTest {

  private static final String BASE_DIR = "src/test/resources/org/sonar/plugins/python";
  private SensorContextTester context = SensorContextTester.create(new File(BASE_DIR));
  private PythonCpdAnalyzer cpdAnalyzer = new PythonCpdAnalyzer(context);

  @Test
  public void code_chunks_2() throws Exception {
    DefaultInputFile inputFile = inputFile("code_chunks_2.py");
    PythonVisitorContext visitorContext = TestPythonVisitorRunner.createContext(inputFile.path().toFile());
    cpdAnalyzer.pushCpdTokens(inputFile, visitorContext);

    List<TokensLine> lines = context.cpdTokens("moduleKey:code_chunks_2.py");
    assertThat(lines).hasSize(25);
    TokensLine line1 = lines.get(0);
    assertThat(line1.getStartLine()).isEqualTo(2);
    assertThat(line1.getEndLine()).isEqualTo(2);
    assertThat(line1.getStartUnit()).isEqualTo(1);
    assertThat(line1.getEndUnit()).isEqualTo(1);
    List<String> values = lines.stream().map(TokensLine::getValue).collect(Collectors.toList());
    assertThat(values).containsExactly(
      "00000",
      "1111L",
      "0x10000L",
      "0X1111",
      "0b1111L",
      "0o12345",
      "u\"lala\"",
      "U\"lala\"",
      "r\"lala\"",
      "R\"lala\"",
      "print",
      "a=[1,",
      "2,",
      "]",
      "deffoo():pass",
      "classbar(object):pass",
      "deffoo2(x,y,z,):",
      "pass",
      "defbar(*baz):",
      "foo(3,4,5)",
      "defbar2(**baz):",
      "yield;",
      "items=[]",
      "(itemforiteminitems)",
      "[itemforiteminitems]");
  }

  private DefaultInputFile inputFile(String fileName) {
    File file = new File(BASE_DIR, fileName);

    DefaultInputFile inputFile = TestInputFileBuilder.create("moduleKey", file.getName())
      .setModuleBaseDir(Paths.get(BASE_DIR))
      .setCharset(UTF_8)
      .setType(InputFile.Type.MAIN)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(file, Charsets.UTF_8))
      .build();

    context.fileSystem().add(inputFile);

    return inputFile;
  }
}
