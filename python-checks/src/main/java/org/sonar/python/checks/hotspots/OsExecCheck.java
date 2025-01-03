/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks.hotspots;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.checks.AbstractCallExpressionCheck;

@Rule(key = OsExecCheck.CHECK_KEY)
public class OsExecCheck extends AbstractCallExpressionCheck {

  public static final String CHECK_KEY = "S4721";
  private static final String MESSAGE = "Make sure that executing this OS command is safe here.";

  private static final Set<String> questionableFunctions = new HashSet<>(Arrays.asList("subprocess.run",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.getstatusoutput",
    "subprocess.getoutput",
    "os.system",
    "os.spawnl",
    "os.spawnle",
    "os.spawnlp",
    "os.spawnlpe",
    "os.spawnv",
    "os.spawnve",
    "os.spawnvp",
    "os.spawnvpe",
    "os.popen",
    "os.startfile",
    "os.execl",
    "os.execle",
    "os.execlp",
    "os.execlpe",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.popen2",
    "os.popen3",
    "os.popen4",
    "popen2.popen2",
    "popen2.popen3",
    "popen2.popen4"
  ));

  @Override
  protected Set<String> functionsToCheck() {
    return questionableFunctions;
  }

  @Override
  protected String message() {
    return MESSAGE;
  }
}
