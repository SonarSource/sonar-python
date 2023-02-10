/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonVersionUtils;
import org.sonar.plugins.python.indexer.PythonIndexer;
import org.sonar.python.checks.CheckList;

import static org.sonar.plugins.python.api.PythonVersionUtils.PYTHON_VERSION_KEY;

public final class IPynbSensor implements Sensor {

  private final PythonChecks checks;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final NoSonarFilter noSonarFilter;
  private final PythonIndexer indexer;

  public IPynbSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter, PythonIndexer indexer) {
    this.checks = new PythonChecks(checkFactory)
      .addChecks("ipython", CheckList.getChecks());
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
    this.indexer = indexer;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .onlyOnLanguage(IPynb.KEY)
      .name("IPython Notebooks Sensor");
  }

  @Override
  public void execute(SensorContext context) {
    List<InputFile> pythonFiles = getInputFiles(context);
    context.config().get(PYTHON_VERSION_KEY)
      .map(PythonVersionUtils::fromString)
      .ifPresent(ProjectPythonVersion::setCurrentVersions);
    PythonScanner scanner = new PythonScanner(context, checks, fileLinesContextFactory, noSonarFilter, indexer);
    scanner.execute(pythonFiles, context);
  }

  private static List<InputFile> getInputFiles(SensorContext context) {
    FilePredicates p = context.fileSystem().predicates();
    Iterable<InputFile> it = context.fileSystem().inputFiles(p.and(p.hasLanguage(IPynb.KEY)));
    List<InputFile> list = new ArrayList<>();
    it.forEach(list::add);
    return Collections.unmodifiableList(list);
  }
}
