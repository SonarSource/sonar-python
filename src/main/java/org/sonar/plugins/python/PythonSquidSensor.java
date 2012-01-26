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

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.measures.CoreMetrics;
import org.sonar.api.resources.InputFile;
import org.sonar.api.resources.ProjectFileSystem;
import org.sonar.api.resources.Project;
import org.sonar.squid.measures.Metric;
import org.sonar.squid.text.Source;

public final class PythonSquidSensor extends PythonSensor {
  protected void analyzeFile(InputFile inputFile, Project project, SensorContext sensorContext) throws IOException {
    // the comment syntax cannot be controlled fully due to poorness of sonar API:
    // the multiline and single-line java syntax are hardcoded, only
    // additional single-line comment syntax can be specified. A better
    // implementation should be possible with one of the future sonar releases

    Reader reader = null;
    try {
      reader = new StringReader(FileUtils.readFileToString(inputFile.getFile(), project.getFileSystem().getSourceCharset().name()));
      org.sonar.api.resources.File pyfile = PythonFile.fromIOFile(inputFile.getFile(), project.getFileSystem().getSourceDirs());
      Source source = new Source(reader, new PythonRecognizer(), new String[] { "#" });

      sensorContext.saveMeasure(pyfile, CoreMetrics.FILES, 1.0);
      sensorContext.saveMeasure(pyfile, CoreMetrics.LINES, (double) source.getMeasure(Metric.LINES));
      sensorContext.saveMeasure(pyfile, CoreMetrics.COMMENT_LINES, (double) source.getMeasure(Metric.COMMENT_LINES));
      sensorContext.saveMeasure(pyfile, CoreMetrics.NCLOC, (double) source.getMeasure(Metric.LINES_OF_CODE));
    } finally {
      IOUtils.closeQuietly(reader);
    }
  }
}
