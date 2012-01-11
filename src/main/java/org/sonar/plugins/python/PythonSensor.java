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

//import java.io.File;
import java.io.IOException;
//import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.batch.Sensor;
import org.sonar.api.batch.SensorContext;
import org.sonar.api.resources.InputFile;
import org.sonar.api.resources.Project;
import org.sonar.api.resources.ProjectFileSystem;

public abstract class PythonSensor implements Sensor {
  private static final Logger LOGGER = LoggerFactory.getLogger(PythonSensor.class);

  public boolean shouldExecuteOnProject(Project project) {
    return Python.KEY.equals(project.getLanguageKey());
  }

  public void analyse(Project project, SensorContext sensorContext) {
    for (InputFile inputFile: project.getFileSystem().mainFiles(Python.KEY)) {
      try {
        analyzeFile(inputFile, project.getFileSystem(), sensorContext);
      } catch (Exception e) {
        LOGGER.error("Cannot analyze the file '{}', details: '{}'", inputFile.getFile().getAbsolutePath(), e);
      }
    }
  }

  protected abstract void analyzeFile(InputFile inputFile, ProjectFileSystem pfs, SensorContext sensorContext) throws IOException;
}
