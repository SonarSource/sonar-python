/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
package org.sonar.plugins.python.pylint;

import com.google.common.io.Files;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.utils.command.Command;
import org.sonar.api.utils.command.CommandExecutor;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.LinkedList;
import java.util.List;

public class PylintIssuesAnalyzer {

  private static final Logger LOG = LoggerFactory.getLogger(PylintSensor.class);

  private static final String FALLBACK_PYLINT = "pylint";

  private String pylint = null;
  private String pylintConfigParam = null;
  private PylintArguments pylintArguments;

  PylintIssuesAnalyzer(String pylintPath, String pylintConfigPath) {
    this(pylintPath, pylintConfigPath, new PylintArguments(Command.create(pylintPathWithDefault(pylintPath)).addArgument("--version")));
  }

  PylintIssuesAnalyzer(String pylintPath, String pylintConfigPath, PylintArguments arguments) {
    pylint = pylintPathWithDefault(pylintPath);

    if (pylintConfigPath != null) {
      if (!new File(pylintConfigPath).exists()) {
        throw new IllegalStateException("Cannot find the pylint configuration file: " + pylintConfigPath);
      }
      pylintConfigParam = "--rcfile=" + pylintConfigPath;
    }

    pylintArguments = arguments;
  }

  private static String pylintPathWithDefault(String pylintPath) {
    if (pylintPath != null) {
      if (!new File(pylintPath).exists()) {
        throw new IllegalStateException("Cannot find the pylint executable: " + pylintPath);
      }
      return pylintPath;
    }
    return FALLBACK_PYLINT;
  }

  public List<Issue> analyze(String path, Charset charset, File out) throws IOException {
    Command command = Command.create(pylint).addArguments(pylintArguments.arguments()).addArgument(path);

    if (pylintConfigParam != null) {
      command.addArgument(pylintConfigParam);
    }

    LOG.debug("Calling command: '{}'", command.toString());

    long timeoutMS = 300000; // =5min
    CommandStreamConsumer stdOut = new CommandStreamConsumer();
    CommandStreamConsumer stdErr = new CommandStreamConsumer();
    CommandExecutor.create().execute(command, stdOut, stdErr, timeoutMS);

    // the error stream can contain a line like 'no custom config found, using default'
    // any bigger output on the error stream is likely a pylint malfunction
    if (stdErr.getData().size() > 1) {
      LOG.warn("Output on the error channel detected: this is probably due to a problem on pylint's side.");
      LOG.warn("Content of the error stream: \n\"{}\"", StringUtils.join(stdErr.getData(), "\n"));
    }

    Files.write(StringUtils.join(stdOut.getData(), "\n"), out, charset);

    return parseOutput(stdOut.getData());
  }

  protected List<Issue> parseOutput(List<String> lines) {
    List<Issue> issues = new LinkedList<Issue>();

    PylintReportParser parser = new PylintReportParser();
    if (!lines.isEmpty()) {
      for (String line : lines) {
        Issue issue = parser.parseLine(line);
        if (issue != null){
          issues.add(issue);
        }
      }
    }

    return issues;
  }
}
