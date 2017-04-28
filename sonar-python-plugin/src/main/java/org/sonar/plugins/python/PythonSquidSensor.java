/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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

import com.google.common.collect.ImmutableList;
import java.util.List;
import org.sonar.api.SonarProduct;
import org.sonar.api.batch.fs.FilePredicates;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.fs.InputFile.Type;
import org.sonar.api.batch.rule.CheckFactory;
import org.sonar.api.batch.rule.Checks;
import org.sonar.api.batch.sensor.Sensor;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.SensorDescriptor;
import org.sonar.api.issue.NoSonarFilter;
import org.sonar.api.measures.FileLinesContextFactory;
import org.sonar.api.utils.Version;
import org.sonar.plugins.python.coverage.PythonCoverageSensor;
import org.sonar.python.PythonCheck;
import org.sonar.python.PythonConfiguration;
import org.sonar.python.checks.CheckList;
import org.sonar.python.metrics.MetricVisitor;

public final class PythonSquidSensor implements Sensor {

  private static final Version V6_0 = Version.create(6, 0);

  private final Checks<PythonCheck> checks;
  private final FileLinesContextFactory fileLinesContextFactory;
  private final NoSonarFilter noSonarFilter;

  public PythonSquidSensor(FileLinesContextFactory fileLinesContextFactory, CheckFactory checkFactory, NoSonarFilter noSonarFilter) {
    this.checks = checkFactory
      .<PythonCheck>create(CheckList.REPOSITORY_KEY)
      .addAnnotatedChecks(CheckList.getChecks());
    this.fileLinesContextFactory = fileLinesContextFactory;
    this.noSonarFilter = noSonarFilter;
  }

  @Override
  public void describe(SensorDescriptor descriptor) {
    descriptor
      .onlyOnLanguage(Python.KEY)
      .name("Python Squid Sensor")
      .onlyOnFileType(Type.MAIN);
  }

  @Override
  public void execute(SensorContext context) {
    PythonConfiguration conf = new PythonConfiguration(context.fileSystem().encoding());

    MetricVisitor metricVisitor = new MetricVisitor(fileLinesContextFactory, context.fileSystem(), conf.getIgnoreHeaderComments());
    FilePredicates p = context.fileSystem().predicates();
    List<InputFile> inputFiles = ImmutableList.copyOf(
      context.fileSystem().inputFiles(p.and(p.hasType(InputFile.Type.MAIN), p.hasLanguage(Python.KEY))));

    new PythonScanner(context, checks, metricVisitor, noSonarFilter, inputFiles).scanFiles();

    if (!isSonarLint(context)) {
      (new PythonCoverageSensor()).execute(context, metricVisitor.fileLinesVisitor().linesOfCodeByFile());
    }
  }

  private static boolean isSonarLint(SensorContext context) {
    return context.getSonarQubeVersion().isGreaterThanOrEqual(V6_0) && context.runtime().getProduct() == SonarProduct.SONARLINT;
  }

}
