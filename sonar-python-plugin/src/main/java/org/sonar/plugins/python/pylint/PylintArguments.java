/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.plugins.python.pylint;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.sonar.api.utils.command.Command;
import org.sonar.api.utils.command.CommandExecutor;

public class PylintArguments {

  private static final Pattern PYLINT_VERSION_PATTERN = Pattern.compile(".*pylint[^ ]* ([0-9\\.]+).*");
  private static final String[] ARGS_PYLINT_0_X = {"-i", "y", "-f", "parseable", "-r", "n"};
  private static final String[] ARGS_PYLINT_1_X = {"--msg-template", "{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}", "-r", "n"};

  private final String[] arguments;

  public PylintArguments(Command command) {
    String pylintVersion = pylintVersion(command);
    this.arguments = pylintVersion.startsWith("0") ? ARGS_PYLINT_0_X : ARGS_PYLINT_1_X;
  }

  private static String pylintVersion(Command command) {
    long timeout = 10_000;
    CommandStreamConsumer out = new CommandStreamConsumer();
    CommandStreamConsumer err = new CommandStreamConsumer();
    CommandExecutor.create().execute(command, out, err, timeout);
    Stream<String> outputLines = Stream.concat(out.getData().stream(), err.getData().stream());

    for (String outLine : (Iterable<String>) outputLines::iterator) {
      Matcher matcher = PYLINT_VERSION_PATTERN.matcher(outLine);
      if (matcher.matches()) {
        return matcher.group(1);
      }
    }
    String message = String.format("Failed to determine pylint version with command: \"%s\", received %d line(s) of output:%n%s",
      command.toCommandLine(), out.getData().size() + err.getData().size(), out.getData() + "\n" + err.getData());
    throw new IllegalArgumentException(message);
  }

  public String[] arguments() {
    return arguments;
  }

}
