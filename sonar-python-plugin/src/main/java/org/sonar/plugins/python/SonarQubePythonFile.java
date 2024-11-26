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

import java.io.IOException;
import java.net.URI;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.api.PythonFile;

public abstract class SonarQubePythonFile implements PythonFile {

  private final InputFile inputFile;

  private SonarQubePythonFile(InputFile inputFile) {
    this.inputFile = inputFile;
  }

  public static PythonFile create(InputFile inputFile) {
    return new Sq62File(inputFile);
  }

  public static PythonFile create(PythonInputFile pythonInputFile) {
    if (pythonInputFile.kind() == PythonInputFile.Kind.PYTHON) {
      return new Sq62File(pythonInputFile.wrappedFile());
    } else {
      try {
        return new IpynbFile(pythonInputFile);
      } catch (IOException e) {
        throw new IllegalStateException("Cannot read " + pythonInputFile, e);
      }
    }
  }

  @Override
  public String fileName() {
    return inputFile.filename();
  }

  public InputFile inputFile() {
    return inputFile;
  }

  @Override
  public URI uri() {
    return inputFile().uri();
  }

  @Override
  public String key() {
    return inputFile().key();
  }

  @Override
  public String toString() {
    return inputFile.toString();
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

  public static class IpynbFile extends SonarQubePythonFile {

    private final String contents;

    private IpynbFile(PythonInputFile inputFile) throws IOException {
      super(inputFile.wrappedFile());
      contents = inputFile.contents();
    }

    @Override
    public String content() {
      return contents;
    }

  }

}
