/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LineOffsetCounter {
  private static final char LINE_FEED = '\n';
  private static final char CARRIAGE_RETURN = '\r';
  private int currentOriginalLineStartOffset = 0;
  private final List<Integer> originalLineStartOffsets = new ArrayList<>();
  private int[] originalLineStartOffsetsArray;


  LineOffsetCounter(String literalValue) {
    originalLineStartOffsets.add(0);
    read(literalValue);
  }

  private void handleAll() {
    currentOriginalLineStartOffset++;
  }

  private void newLine() {
    originalLineStartOffsets.add(currentOriginalLineStartOffset);
  }

  int findLine(int offset) {
    return Math.abs(Arrays.binarySearch(startOffset(), offset) + 1);
  }

  int findColumn(int line, int offset) {
    return offset - startOffset()[line - 1];
  }

  private int[] startOffset() {
    if (originalLineStartOffsetsArray == null) {
      originalLineStartOffsetsArray = originalLineStartOffsets.stream().mapToInt(i -> i).toArray();
    }
    return originalLineStartOffsetsArray;
  }

  private void read(String literalValue) {
    boolean afterCR = false;
    for (char c : literalValue.toCharArray()) {
      if (afterCR) {
        if (c == LINE_FEED) {
          handleAll();
          newLine();
        } else {
          newLine();
          handleAll();
        }
        afterCR = c == CARRIAGE_RETURN;
      } else if (c == LINE_FEED) {
        handleAll();
        newLine();
      } else if (c == CARRIAGE_RETURN) {
        afterCR = true;
        handleAll();
      } else {
        handleAll();
      }
    }
    if (afterCR) {
      newLine();
    }
  }

}
