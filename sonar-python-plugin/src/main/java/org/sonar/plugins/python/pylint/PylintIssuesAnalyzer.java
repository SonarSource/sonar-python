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
package org.sonar.plugins.python.pylint;

import com.google.common.collect.ImmutableMap;
import com.google.common.io.Files;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.utils.SonarException;
import org.sonar.api.utils.command.Command;
import org.sonar.api.utils.command.CommandExecutor;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PylintIssuesAnalyzer {

  private static final Logger LOG = LoggerFactory.getLogger(PylintSensor.class);

  // Pylint 0.24 brings a nasty reidentifying of some rules...
  // To avoid burdening of users with rule clones we map the ids.
  // This workaround can die as soon as pylints <= 0.23.X become obsolete.
  private static final Map<String, String> ID_MAP = ImmutableMap.<String, String>builder()
    .put("E9900", "E1300")
    .put("E9901", "E1301")
    .put("E9902", "E1302")
    .put("E9903", "E1303")
    .put("E9904", "E1304")
    .put("E9905", "E1305")
    .put("E9906", "E1306")
    .put("W6501", "W1201")
    .put("W9900", "W1300")
    .put("W9901", "W1301")
    .build();

  private static final String FALLBACK_PYLINT = "pylint";
  private static final Pattern PATTERN = Pattern.compile("(.+):([0-9]+): \\[(.*)\\] (.*)");

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
        throw new SonarException("Cannot find the pylint configuration file: " + pylintConfigPath);
      }
      pylintConfigParam = "--rcfile=" + pylintConfigPath;
    }
    
    pylintArguments = arguments;
  }

  private static String pylintPathWithDefault(String pylintPath) {
    if (pylintPath != null) {
      if (!new File(pylintPath).exists()) {
        throw new SonarException("Cannot find the pylint executable: " + pylintPath);
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
    // Parse the output of pylint. Example of the format:
    //
    // complexity/code_chunks.py:62: [W0104, list_compr] Statement seems to have no effect
    // complexity/code_chunks.py:64: [C0111, list_compr_filter] Missing docstring
    // ...

    List<Issue> issues = new LinkedList<Issue>();

    int linenr;
    String filename = null;
    String ruleid = null;
    String objname = null;
    String descr = null;

    if (!lines.isEmpty()) {
      for (String line : lines) {
        if (line.length() > 0) {
          if (!isDetail(line)) {
            Matcher m = PATTERN.matcher(line);
            if (m.matches() && m.groupCount() == 4) {
              filename = m.group(1);
              linenr = Integer.valueOf(m.group(2));
              String[] parts = m.group(3).split(",");

              ruleid = ruleId(parts[0].trim());

              if (parts.length == 2) {
                objname = parts[1].trim();
              }

              descr = m.group(4);
              issues.add(new Issue(filename, linenr, ruleid, objname, descr));
            } else {
              LOG.debug("Cannot parse the line: {}", line);
            }
          } else {
            LOG.trace("Classifying as detail and ignoring line '{}'", line);
          }
        }
      }
    }

    return issues;
  }

  private String ruleId(String ruleAndMessageIds) {
    String ruleid = ruleAndMessageIds;
    int parenthesisIndex = ruleid.indexOf('(');
    if (parenthesisIndex > -1) {
      ruleid = ruleid.substring(0, parenthesisIndex);
    }
    if (ID_MAP.containsKey(ruleid)) {
      ruleid = ID_MAP.get(ruleid);
    }
    return ruleid;
  }

  private boolean isDetail(String line) {
    char first = line.charAt(0);
    return first == ' ' || first == '\t' || first == '\n';
  }

}
