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
package org.sonar.plugins.python.cpd;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.sensor.cpd.internal.TokensLine;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.plugins.python.Python;
import org.sonar.plugins.python.TestUtils;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.TestPythonVisitorRunner;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.assertj.core.api.Assertions.assertThat;

class PythonCpdAnalyzerTest {

  private static final String BASE_DIR = "src/test/resources/org/sonar/plugins/python";
  private SensorContextTester context = SensorContextTester.create(new File(BASE_DIR));
  private PythonCpdAnalyzer cpdAnalyzer = new PythonCpdAnalyzer(context);

  @Test
  void code_chunks_2() {
    DefaultInputFile inputFile = inputFile("code_chunks_2.py");
    PythonVisitorContext visitorContext = TestPythonVisitorRunner.createContext(inputFile.path().toFile());
    cpdAnalyzer.pushCpdTokens(inputFile, visitorContext);

    List<TokensLine> lines = context.cpdTokens("moduleKey:code_chunks_2.py");
    assertThat(lines).isNotNull().hasSize(29);
    TokensLine line1 = lines.get(0);
    assertThat(line1.getStartLine()).isEqualTo(2);
    assertThat(line1.getEndLine()).isEqualTo(2);
    assertThat(line1.getStartUnit()).isEqualTo(1);
    assertThat(line1.getEndUnit()).isEqualTo(1);
    List<String> values = lines.stream().map(TokensLine::getValue).toList();
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
      // \n materializes DEDENT
      "pass\n",
      "defbar(*baz):",
      // \n materializes DEDENT
      "foo(3,4,5)\n",
      "defbar2(**baz):",
      // \n materializes DEDENT
      "yield;\n",
      "items=[]",
      "(itemforiteminitems)",
      "[itemforiteminitems]",
      "ifitemisnotNone:",
      "pass\n",
      "ifitemnotinitems:",
      "pass\n");
  }

  @Test
  void dedent_with_cpd() {
    DefaultInputFile inputFile = inputFile("cpd_dedent.py");
    PythonVisitorContext visitorContext = TestPythonVisitorRunner.createContext(inputFile.path().toFile());
    cpdAnalyzer.pushCpdTokens(inputFile, visitorContext);
    List<TokensLine> tokensLines = context.cpdTokens("moduleKey:cpd_dedent.py");
    assertThat(tokensLines).isNotNull();
    assertThat(tokensLines.size() % 2).isEqualTo(0);
    int mid = tokensLines.size() / 2;
    for (int i = 0; i < mid; i++) {
      TokensLine tokensLine = tokensLines.get(i);
      TokensLine dup = tokensLines.get(mid + i);
      assertThat(tokensLine.getStartLine() + 5).isEqualTo(dup.getStartLine());
      if (dup.getStartLine() == 9) {
        // line 9 contains a different DEDENT and thus the token values should not be equal
        assertThat(tokensLine.getValue()).isNotEqualTo(dup.getValue());
      } else {
        assertThat(tokensLine.getValue()).isEqualTo(dup.getValue());
      }
    }
  }

  private DefaultInputFile inputFile(String fileName) {
    File file = new File(BASE_DIR, fileName);

    DefaultInputFile inputFile = TestInputFileBuilder.create("moduleKey", file.getName())
      .setModuleBaseDir(Paths.get(BASE_DIR))
      .setCharset(UTF_8)
      .setType(InputFile.Type.MAIN)
      .setLanguage(Python.KEY)
      .initMetadata(TestUtils.fileContent(file, StandardCharsets.UTF_8))
      .build();

    context.fileSystem().add(inputFile);

    return inputFile;
  }
}
