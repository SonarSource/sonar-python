/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
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
package org.sonar.plugins.python.flake8;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.utils.SonarException;
import org.sonar.api.utils.command.Command;
import org.sonar.api.utils.command.CommandExecutor;

import com.google.common.io.Files;

public class Flake8IssuesAnalyzer {

  private static final Logger LOG = LoggerFactory.getLogger(Flake8Sensor.class);


  private static final String FALLBACK_COMMAND = "flake8";
  private static final Pattern PATTERN = Pattern.compile("([^:]+):([0-9]+):([0-9]+): (\\S+) (.*)");

  private String flake8 = null;
  private String flake8ConfigParam = null;

  Flake8IssuesAnalyzer(String flake8Path, String flake8ConfigPath) {
    flake8 = flake8PathWithDefault(flake8Path);

    if (flake8ConfigPath != null) {
      if (!new File(flake8ConfigPath).exists()) {
        throw new SonarException("Cannot find the flake8 configuration file: " + flake8ConfigPath);
      }
      flake8ConfigParam = "--config=" + flake8ConfigPath;
    }
    
  }

  private static String flake8PathWithDefault(String flake8Path) {
    if (flake8Path != null) {
      if (!new File(flake8Path).exists()) {
        throw new SonarException("Cannot find the flake8 executable: " + flake8Path);
      }
      return flake8Path;
    }
    return FALLBACK_COMMAND;
  }

  public List<Issue> analyze(String path, Charset charset, File out) throws IOException {
    Command command = Command.create(flake8).addArgument(path);

    if (flake8ConfigParam != null) {
      command.addArgument(flake8ConfigParam);
    }

    LOG.debug("Calling command: '{}'", command.toString());

    long timeoutMS = 300000; // =5min
    CommandStreamConsumer stdOut = new CommandStreamConsumer();
    CommandStreamConsumer stdErr = new CommandStreamConsumer();
    CommandExecutor.create().execute(command, stdOut, stdErr, timeoutMS);

    // the error stream can contain a line like 'no custom config found, using default'
    // any bigger output on the error stream is likely a flake8 malfunction
    if (stdErr.getData().size() > 1) {
      LOG.warn("Output on the error channel detected: this is probably due to a problem on flake8's side.");
      LOG.warn("Content of the error stream: \n\"{}\"", StringUtils.join(stdErr.getData(), "\n"));
    }

    Files.write(StringUtils.join(stdOut.getData(), "\n"), out, charset);

    return parseOutput(stdOut.getData());
  }

  protected List<Issue> parseOutput(List<String> lines) {
    // Parse the output of flake8. Example of the format:
    //
    // complexity/code_chunks.py:62:32: W0104 Statement seems to have no effect

    List<Issue> issues = new LinkedList<Issue>();

    int linenr;
    String filename = null;
    String ruleid = null;
    String descr = null;

    if (!lines.isEmpty()) {
      for (String line : lines) {
        if (line.length() > 0) {
            Matcher m = PATTERN.matcher(line);
            if (m.matches() && m.groupCount() == 5) {
              filename = m.group(1);
              linenr = Integer.valueOf(m.group(2));
              ruleid = m.group(4);
              descr = m.group(5);
              issues.add(new Issue(filename, linenr, ruleid, descr));
            } else {
              LOG.debug("Cannot parse the line: {}", line);
            }
          } else {
            LOG.trace("Classifying as detail and ignoring line '{}'", line);
          }
        }
    }

    return issues;
  }

}
