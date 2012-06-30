/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */

package org.sonar.plugins.python;

import com.sonar.sslr.squid.AstScanner;
import org.sonar.api.batch.Sensor;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.resources.File;
import org.sonar.api.resources.InputFileUtils;
import org.sonar.api.resources.Project;
import org.sonar.python.PythonAstScanner;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonMetric;
import org.sonar.squid.api.SourceCode;
import org.sonar.squid.api.SourceFile;
import org.sonar.squid.indexer.QueryByType;

import java.util.Collection;

public final class PythonSquidSensor implements Sensor {

  private Project project;
  private SensorContext context;
  private AstScanner<PythonGrammar> scanner;

  public boolean shouldExecuteOnProject(Project project) {
    return Python.KEY.equals(project.getLanguageKey());
  }

  public void analyse(Project project, SensorContext context) {
    this.project = project;
    this.context = context;

    this.scanner = PythonAstScanner.create(createConfiguration(project));
    // Collection<SquidCheck> squidChecks = annotationCheckFactory.getChecks();
    // this.scanner = PythonAstScanner.create(createConfiguration(project), squidChecks.toArray(new SquidCheck[squidChecks.size()]));
    scanner.scanFiles(InputFileUtils.toFiles(project.getFileSystem().mainFiles(Python.KEY)));

    Collection<SourceCode> squidSourceFiles = scanner.getIndex().search(new QueryByType(SourceFile.class));
    save(squidSourceFiles);
  }

  private PythonConfiguration createConfiguration(Project project) {
    return new PythonConfiguration(project.getFileSystem().getSourceCharset());
  }

  private void save(Collection<SourceCode> squidSourceFiles) {
    for (SourceCode squidSourceFile : squidSourceFiles) {
      SourceFile squidFile = (SourceFile) squidSourceFile;

      File sonarFile = File.fromIOFile(new java.io.File(squidFile.getKey()), project);

      // saveFilesComplexityDistribution(sonarFile, squidFile);
      // saveFunctionsComplexityDistribution(sonarFile, squidFile);
      saveMeasures(sonarFile, squidFile);
      // saveViolations(sonarFile, squidFile);
    }
  }

  private void saveMeasures(File sonarFile, SourceFile squidFile) {
    context.saveMeasure(sonarFile, CoreMetrics.FILES, squidFile.getDouble(PythonMetric.FILES));
    context.saveMeasure(sonarFile, CoreMetrics.LINES, squidFile.getDouble(PythonMetric.LINES));
    context.saveMeasure(sonarFile, CoreMetrics.NCLOC, squidFile.getDouble(PythonMetric.LINES_OF_CODE));
    // context.saveMeasure(sonarFile, CoreMetrics.FUNCTIONS, squidFile.getDouble(PythonMetric.FUNCTIONS));
    // context.saveMeasure(sonarFile, CoreMetrics.STATEMENTS, squidFile.getDouble(PythonMetric.STATEMENTS));
    // context.saveMeasure(sonarFile, CoreMetrics.COMPLEXITY, squidFile.getDouble(PythonMetric.COMPLEXITY));
    context.saveMeasure(sonarFile, CoreMetrics.COMMENT_BLANK_LINES, squidFile.getDouble(PythonMetric.COMMENT_BLANK_LINES));
    context.saveMeasure(sonarFile, CoreMetrics.COMMENT_LINES, squidFile.getDouble(PythonMetric.COMMENT_LINES));
  }

  @Override
  public String toString() {
    return getClass().getSimpleName();
  }

}
