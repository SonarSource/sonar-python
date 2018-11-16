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
package org.sonar.plugins.python;

import java.io.IOException;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.python.PythonFile;

public abstract class SonarQubePythonFile implements PythonFile {

  private final InputFile inputFile;

  private SonarQubePythonFile(InputFile inputFile) {
    this.inputFile = inputFile;
  }

  public static PythonFile create(InputFile inputFile) {
    return new Sq62File(inputFile);
  }

  @Override
  public String fileName() {
    return inputFile.path().getFileName().toString();
  }

  public InputFile inputFile() {
    return inputFile;
  }

  private static class Sq62File extends SonarQubePythonFile {

    public Sq62File(InputFile inputFile) {
      super(inputFile);
    }

    @Override
    public String content() {
      try {
        return inputFile().contents();
      } catch (IOException e) {
        throw new IllegalStateException("Could not read content of input file " + inputFile(), e);
      }
    }

  }

}
