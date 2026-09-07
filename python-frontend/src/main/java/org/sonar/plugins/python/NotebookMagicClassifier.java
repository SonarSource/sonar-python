/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python;

import java.util.Set;

public final class NotebookMagicClassifier {

  private static final Set<String> DATABRICKS_OPAQUE_CELL_MAGICS = Set.of("sql", "scala", "r", "md", "md-sandbox", "sh", "fs", "run", "skip");

  private NotebookMagicClassifier() {
  }

  public static boolean isOpaqueCell(NotebookDialect dialect, CharSequence source) {
    if (source.isEmpty() || source.charAt(0) != '%') {
      return false;
    }
    if (source.length() > 1 && source.charAt(1) == '%') {
      return true;
    }
    if (dialect != NotebookDialect.DATABRICKS) {
      return false;
    }

    int commandEnd = 1;
    while (commandEnd < source.length() && !Character.isWhitespace(source.charAt(commandEnd))) {
      commandEnd++;
    }
    String command = source.subSequence(1, commandEnd).toString();
    return DATABRICKS_OPAQUE_CELL_MAGICS.contains(command);
  }
}
