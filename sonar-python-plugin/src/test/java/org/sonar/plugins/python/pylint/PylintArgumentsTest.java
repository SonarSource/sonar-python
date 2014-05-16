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

import org.junit.Ignore;
import org.junit.Test;
import org.sonar.api.utils.command.Command;

import static org.fest.assertions.Assertions.assertThat;

@Ignore
public class PylintArgumentsTest {

  @Test
  public void pylint_0_x() {
    String[] arguments = new PylintArguments(Command.create("echo").addArgument("pylint 0.28.0")).arguments();
    assertThat(arguments).containsOnly("-i", "y", "-f", "parseable", "-r", "n");
  }

  @Test
  public void pylint_1_x() {
    String[] arguments = new PylintArguments(Command.create("echo").addArgument("pylint 1.1.0")).arguments();
    assertThat(arguments).containsOnly("--msg-template", "{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}", "-r", "n");
  }

  @Test(expected = IllegalArgumentException.class)
  public void unknown() throws Exception {
    new PylintArguments(Command.create("echo"));
  }

}
