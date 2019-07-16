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
package org.sonar.plugins.python.minimization;

import com.intellij.psi.util.PsiTreeUtil;
import com.jetbrains.python.psi.PyFile;
import com.jetbrains.python.psi.PyFormattedStringElement;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.Collectors;
import org.mockito.Mockito;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.internal.DefaultInputFile;
import org.sonar.api.batch.fs.internal.TestInputFileBuilder;
import org.sonar.api.batch.rule.Checks;
import org.sonar.api.batch.sensor.internal.SensorContextTester;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.plugins.python.PythonScanner;
import org.sonar.plugins.python.TestUtils;
import org.sonar.python.PythonCheck;
import org.sonar.python.frontend.PythonParser;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * To be executed in the IDE with the following JVM option: -verbose:class
 */
public class ParserCaller {

  private final PythonParser parser = new PythonParser();

  public static void main(String[] args) {
    ParserCaller test = new ParserCaller();
    test.python3_f_string();
    test.real_files();
  }

  private void python3_f_string() {
    PyFile pyFile = parser.parse("f\"Hello {name}!\"");
    PyFormattedStringElement stringElement = PsiTreeUtil.getParentOfType(pyFile.findElementAt(0), PyFormattedStringElement.class);
    check(stringElement.getDecodedFragments().stream().map(pair -> pair.second).collect(Collectors.toList()).equals(Arrays.asList("Hello ", "{name}", "!")));
  }

  private void real_files() {
    parseFile("its/sources/buildbot-0.8.6p1/buildbot/changes/p4poller.py");
    parseFile("its/sources/buildbot-0.8.6p1/buildbot/buildrequest.py");
    parseFile("its/sources/buildbot-0.8.6p1/buildbot/status/web/builder.py");
  }

  private void parseFile(String path) {
    File file = new File(path);
    DefaultInputFile inputFile = TestInputFileBuilder.create(".", file.getPath())
      .setCharset(StandardCharsets.UTF_8)
      .initMetadata(TestUtils.fileContent(file, StandardCharsets.UTF_8))
      .build();
    SensorContextTester context = SensorContextTester.create(new File("."));
    context.fileSystem().add(inputFile);
    Checks<PythonCheck> checks = Mockito.mock(Checks.class);
    FileLinesContextFactory fileLinesContextFactory = mock(FileLinesContextFactory.class);
    FileLinesContext fileLinesContext = mock(FileLinesContext.class);
    when(fileLinesContextFactory.createFor(Mockito.any(InputFile.class))).thenReturn(fileLinesContext);
    PythonScanner scanner = new PythonScanner(context, checks, fileLinesContextFactory, new NoSonarFilter(), Collections.singletonList(inputFile));
    scanner.scanFiles();
  }

  private void check(boolean expr) {
    if (!expr) {
      throw new IllegalStateException("Fail!");
    }
  }
}
