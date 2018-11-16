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

import org.apache.commons.lang.SystemUtils;
import org.junit.Test;
import org.sonar.api.utils.command.Command;

import static org.assertj.core.api.Assertions.assertThat;

public class PylintArgumentsTest {

  @Test
  public void pylint_0_x() {
    String[] arguments = new PylintArguments(command("pylint 0.28.0")).arguments();
    assertThat(arguments).containsOnly("-i", "y", "-f", "parseable", "-r", "n");
  }

  @Test
  public void pylint_bat_0_x() {
    String[] arguments = new PylintArguments(command("pylint.bat 0.28.0")).arguments();
    assertThat(arguments).containsOnly("-i", "y", "-f", "parseable", "-r", "n");
  }

  @Test
  public void pylint_1_x() {
    String[] arguments = new PylintArguments(command("pylint 1.1.0")).arguments();
    assertThat(arguments).containsOnly("--msg-template", "{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}", "-r", "n");
  }

  @Test(expected = IllegalArgumentException.class)
  public void unknown() throws Exception {
    new PylintArguments(command(""));
  }

  private Command command(String toOutput) {
    if (SystemUtils.IS_OS_WINDOWS) {
      return Command.create("cmd.exe").addArguments(new String[] {"/c", "echo", toOutput});
    }
    return Command.create("echo").addArgument(toOutput);
  }

}
