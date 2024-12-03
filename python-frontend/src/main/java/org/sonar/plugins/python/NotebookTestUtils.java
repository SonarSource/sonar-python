/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python;

import java.util.List;
import java.util.Map;
import org.sonar.python.EscapeCharPositionInfo;

public class NotebookTestUtils {

  private NotebookTestUtils() {
    // Utility class
  }

  public static List<EscapeCharPositionInfo> mapToColumnMappingList(Map<Integer, Integer> map) {
    return map.entrySet().stream()
      .sorted(Map.Entry.comparingByKey())
      .map(entry -> new EscapeCharPositionInfo(entry.getKey(), entry.getValue()))
      .toList();
  }
}
