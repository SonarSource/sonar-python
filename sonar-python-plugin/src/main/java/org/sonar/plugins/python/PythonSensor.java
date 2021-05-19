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
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
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
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.plugins.python.indexer.SonarQubePythonIndexer;
import org.sonar.python.checks.CheckList;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.tree.PythonTreeMaker;
import org.sonarsource.performance.measure.PerformanceMeasure;

public final class PythonSensor implements Sensor {

  private static final String PERFORMANCE_MEASURE_PROPERTY = "sonar.python.performance.measure";
  private static final String PERFORMANCE_MEASURE_FILE_PATH_PROPERTY = "sonar.python.performance.measure.path";
  private static final String PERFORMANCE_MEASURE_DESTINATION_FILE = "sonar-python-performance-measure.json";

  private final PythonChecks checks;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final NoSonarFilter noSonarFilter;
  private final PythonIndexer indexer;

  /**
   * Constructor to be used by pico if neither PythonCustomRuleRepository nor PythonIndexer are to be found and injected.
   */
  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter) {
    this(fileLinesContextFactory, checkFactory, noSonarFilter, null, null);
  }

  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter,
                      PythonIndexer indexer) {
    this(fileLinesContextFactory, checkFactory, noSonarFilter, null, indexer);
  }

  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter,
                      PythonCustomRuleRepository[] customRuleRepositories) {
    this(fileLinesContextFactory, checkFactory, noSonarFilter, customRuleRepositories, null);
  }

  public PythonSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter,
                      @Nullable PythonCustomRuleRepository[] customRuleRepositories, @Nullable PythonIndexer indexer) {
    this.checks = new PythonChecks(checkFactory)
      .addChecks(CheckList.REPOSITORY_KEY, CheckList.getChecks())
      .addCustomChecks(customRuleRepositories);
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
    this.indexer = indexer;
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
    PerformanceMeasure.Duration durationReport = createPerformanceMeasureReport(context);
    List<InputFile> mainFiles = getInputFiles(Type.MAIN, context);
    List<InputFile> testFiles = getInputFiles(Type.TEST, context);
    PythonIndexer pythonIndexer = this.indexer != null ? this.indexer : new SonarQubePythonIndexer(mainFiles);
    PythonScanner scanner = new PythonScanner(context, checks, fileLinesContextFactory, noSonarFilter, pythonIndexer);
    scanner.execute(mainFiles, context);
    durationReport.stop();
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

  private static PerformanceMeasure.Duration createPerformanceMeasureReport(SensorContext context) {
    return PerformanceMeasure.reportBuilder()
      .activate(context.config().getBoolean(PERFORMANCE_MEASURE_PROPERTY).orElse(Boolean.FALSE))
      .toFile(context.config().get(PERFORMANCE_MEASURE_FILE_PATH_PROPERTY)
        .filter(path -> !path.isEmpty())
        .orElseGet(() -> Optional.ofNullable(context.fileSystem().workDir())
          .filter(File::exists)
          .map(file -> file.toPath().resolve(PERFORMANCE_MEASURE_DESTINATION_FILE).toString())
          .orElse(null)))
      .appendMeasurementCost()
      .start("PythonSensor");
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
