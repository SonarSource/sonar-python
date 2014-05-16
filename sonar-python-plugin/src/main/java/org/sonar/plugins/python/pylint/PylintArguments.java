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

import org.sonar.api.utils.command.Command;
import org.sonar.api.utils.command.CommandExecutor;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PylintArguments {

  private static final Pattern PYLINT_VERSION_PATTERN = Pattern.compile(".*pylint ([0-9\\.]+).*");
  private static final String[] ARGS_PYLINT_0_X = {"-i", "y", "-f", "parseable", "-r", "n"};
  private static final String[] ARGS_PYLINT_1_X = {"--msg-template", "{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}", "-r", "n"};

  private final String[] arguments;

  public PylintArguments(Command command) {
    String pylintVersion = pylintVersion(command);
    this.arguments = pylintVersion.startsWith("0") ? ARGS_PYLINT_0_X : ARGS_PYLINT_1_X;
  }

  private static String pylintVersion(Command command) {
    command.addArgument("--version");
    long timeout = 10000;
    CommandStreamConsumer out = new CommandStreamConsumer();
    CommandStreamConsumer err = new CommandStreamConsumer();
    CommandExecutor.create().execute(command, out, err, timeout);
    for (String outLine : out.getData()) {
      Matcher matcher = PYLINT_VERSION_PATTERN.matcher(outLine);
      if (matcher.matches()) {
        return matcher.group(1);
      }
    }
    throw new IllegalArgumentException("Failed to determine pylint version with command: " + command.toCommandLine());
  }

  public String[] arguments() {
    return arguments;
  }

}
