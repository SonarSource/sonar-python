/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python;

import java.util.Map;
import org.sonar.api.batch.fs.InputFile;

public class GeneratedIPythonFile implements PythonInputFile {

  InputFile wrappedFile;

  Map<Integer, Offset> offsetMap;
  // A map from the line to a Map of column location of escaped chars: 
  // from the position in the Python code to the position in the original file
  Map<Integer, Map<Integer, Integer>> colOffSet;

  public GeneratedIPythonFile(InputFile wrappedFile, Map<Integer, Offset> offsetMap, Map<Integer, Map<Integer, Integer>> colOffSet) {
    this.wrappedFile = wrappedFile;
    this.offsetMap = offsetMap;
    this.colOffSet = colOffSet;
  }

  public Map<Integer, Offset> getOffsetMap() {
    return offsetMap;
  }

  public Map<Integer, Map<Integer, Integer>> getColOffSet() {
    return colOffSet;
  }

  public record Offset(int line, int column) {
  }

  public InputFile wrappedFile() {
    return this.wrappedFile;
  }

  @Override
  public String toString() {
    return wrappedFile.toString();
  }

  @Override
  public boolean equals(Object obj) {
    return wrappedFile.equals(obj);
  }

  @Override
  public int hashCode() {
    return wrappedFile.hashCode();
  }
}
