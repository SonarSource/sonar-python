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

import java.util.Set;
import java.util.concurrent.locks.Lock;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.measure.NewMeasure;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.measures.FileLinesContext;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.measures.Metric;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.metrics.FileLinesVisitor;
import org.sonar.python.metrics.FileMetrics;

public class MeasuresRepository {
  private final SensorContext context;
  private final NoSonarFilter noSonarFilter;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final boolean isInSonarLint;
  private final Lock lock;

  public MeasuresRepository(SensorContext context, NoSonarFilter noSonarFilter, FileLinesContextFactory fileLinesContextFactory, boolean isInSonarLint, Lock lock) {
    this.context = context;
    this.noSonarFilter = noSonarFilter;
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.isInSonarLint = isInSonarLint;
    this.lock = lock;
  }

  public void save(PythonInputFile inputFile, PythonVisitorContext visitorContext) {
    try {
      lock.lock();
      saveInternal(inputFile, visitorContext);
    } finally {
      lock.unlock();
    }
  }

  private void saveInternal(PythonInputFile inputFile, PythonVisitorContext visitorContext) {
    FileMetrics fileMetrics = new FileMetrics(visitorContext, isNotebook(inputFile));
    FileLinesVisitor fileLinesVisitor = fileMetrics.fileLinesVisitor();

    processNoSonarInFile(inputFile, fileLinesVisitor);

    if (!isInSonarLint) {
      Set<Integer> linesOfCode = fileLinesVisitor.getLinesOfCode();
      saveMetricOnFile(inputFile, CoreMetrics.NCLOC, linesOfCode.size());
      saveMetricOnFile(inputFile, CoreMetrics.STATEMENTS, fileMetrics.numberOfStatements());
      saveMetricOnFile(inputFile, CoreMetrics.FUNCTIONS, fileMetrics.numberOfFunctions());
      saveMetricOnFile(inputFile, CoreMetrics.CLASSES, fileMetrics.numberOfClasses());
      saveMetricOnFile(inputFile, CoreMetrics.COMPLEXITY, fileMetrics.complexity());
      saveMetricOnFile(inputFile, CoreMetrics.COGNITIVE_COMPLEXITY, fileMetrics.cognitiveComplexity());
      saveMetricOnFile(inputFile, CoreMetrics.COMMENT_LINES, fileLinesVisitor.getCommentLineCount());

      FileLinesContext fileLinesContext = fileLinesContextFactory.createFor(inputFile.wrappedFile());
      if (inputFile.kind() == PythonInputFile.Kind.PYTHON) {
        for (int line : linesOfCode) {
          fileLinesContext.setIntValue(CoreMetrics.NCLOC_DATA_KEY, line, 1);
        }
      }
      for (int line : fileLinesVisitor.getExecutableLines()) {
        fileLinesContext.setIntValue(CoreMetrics.EXECUTABLE_LINES_DATA_KEY, line, 1);
      }
      save(fileLinesContext);
    }
  }

  private void processNoSonarInFile(PythonInputFile inputFile, FileLinesVisitor fileLinesVisitor) {
    try {
      lock.lock();
      noSonarFilter.noSonarInFile(inputFile.wrappedFile(), fileLinesVisitor.getLinesWithNoSonar());
    } finally {
      lock.unlock();
    }
  }

  static boolean isNotebook(PythonInputFile inputFile) {
    return inputFile.kind() == PythonInputFile.Kind.IPYTHON;
  }

  private void saveMetricOnFile(PythonInputFile inputFile, Metric<Integer> metric, Integer value) {
    var measure = context.<Integer>newMeasure()
      .withValue(value)
      .forMetric(metric)
      .on(inputFile.wrappedFile());
    save(measure);
  }

  private void save(FileLinesContext fileLinesContext) {
    try {
      lock.lock();
      fileLinesContext.save();
    } finally {
      lock.unlock();
    }
  }

  private void save(NewMeasure<Integer> measure) {
    try {
      lock.lock();
      measure.save();
    } finally {
      lock.unlock();
    }
  }
}
