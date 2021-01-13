/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.RecognitionException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonar.plugins.python.api.PythonCustomRuleRepository;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.checks.CheckList;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.PythonTreeMaker;

public final class PythonSensor implements Sensor {

  private final PythonChecks checks;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final NoSonarFilter noSonarFilter;

  /**
   * Constructor to be used by pico if no PythonCustomRuleRepository are to be found and injected.
   */
  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter) {
    this(fileLinesContextFactory, checkFactory, noSonarFilter, null);
  }

  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter,
                      @Nullable PythonCustomRuleRepository[] customRuleRepositories) {
    this.checks = new PythonChecks(checkFactory)
      .addChecks(CheckList.REPOSITORY_KEY, CheckList.getChecks())
      .addCustomChecks(customRuleRepositories);
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .onlyOnLanguage(Python.KEY)
      .name("Python Sensor")
      .onlyOnFileType(Type.MAIN);
  }

  @Override
  public void execute(SensorContext context) {
    List<InputFile> mainFiles = getInputFiles(Type.MAIN, context);
    List<InputFile> testFiles = getInputFiles(Type.TEST, context);
    PythonScanner scanner = new PythonScanner(context, checks, fileLinesContextFactory, noSonarFilter, mainFiles);
    scanner.execute(mainFiles, context);
    if (!testFiles.isEmpty()) {
      new TestHighlightingScanner(context).execute(testFiles, context);
    }
  }

  private static List<InputFile> getInputFiles(InputFile.Type type, SensorContext context) {
    FilePredicates p = context.fileSystem().predicates();
    Iterable<InputFile> it = context.fileSystem().inputFiles(p.and(p.hasType(type), p.hasLanguage(Python.KEY)));
    List<InputFile> list = new ArrayList<>();
    it.forEach(list::add);
    return Collections.unmodifiableList(list);
  }

  private static class TestHighlightingScanner extends Scanner {

    private static final Logger LOG = Loggers.get(TestHighlightingScanner.class);
    private final PythonParser parser = PythonParser.create();

    TestHighlightingScanner(SensorContext context) {
      super(context);
    }

    @Override
    protected String name() {
      return "test sources highlighting";
    }

    @Override
    protected void scanFile(InputFile inputFile) throws IOException {
      try {
        PythonFile pythonFile = SonarQubePythonFile.create(inputFile);
        AstNode astNode = parser.parse(pythonFile.content());
        FileInput parse = new PythonTreeMaker().fileInput(astNode);
        // omitting package and symbols info as it's not required for highlighting
        PythonVisitorContext visitorContext = new PythonVisitorContext(parse, pythonFile, context.fileSystem().workDir(), "", ProjectLevelSymbolTable.empty());
        new PythonHighlighter(context, inputFile).scanFile(visitorContext);
      } catch (RecognitionException e) {
        LOG.error("Unable to parse file: " + inputFile.toString());
        LOG.error(e.getMessage());
      }
    }

    @Override
    protected void processException(Exception e, InputFile file) {
      LOG.warn("Unable to highlight test file: " + file.toString(), e);
    }
  }
}
