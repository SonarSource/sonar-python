/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2017 SonarSource SA
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

import com.google.common.io.Files;
import java.io.IOException;
import java.nio.charset.Charset;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.utils.Version;
import org.sonar.python.PythonFile;

public abstract class SonarQubePythonFile implements PythonFile {

  private static final Version V6_0 = Version.create(6, 0);
  private static final Version V6_2 = Version.create(6, 2);

  private final InputFile inputFile;

  private SonarQubePythonFile(InputFile inputFile) {
    this.inputFile = inputFile;
  }

  public static PythonFile create(InputFile inputFile, SensorContext context) {
    Version version = context.getSonarQubeVersion();
    if (version.isGreaterThanOrEqual(V6_2)) {
      return new Sq62File(inputFile);
    }
    if (version.isGreaterThanOrEqual(V6_0)) {
      return new Sq60File(inputFile);
    }
    return new Sq56File(inputFile, context.fileSystem().encoding());
  }

  @Override
  public String fileName() {
    return inputFile.file().getName();
  }

  public InputFile inputFile() {
    return inputFile;
  }

  private static String contentForCharset(InputFile inputFile, Charset charset) {
    try {
      return Files.toString(inputFile.file(), charset);
    } catch (IOException e) {
      throw new IllegalStateException("Could not read content of input file " + inputFile, e);
    }
  }

  private static class Sq56File extends SonarQubePythonFile {

    private final Charset fileSystemEncoding;

    public Sq56File(InputFile inputFile, Charset fileSystemEncoding) {
      super(inputFile);
      this.fileSystemEncoding = fileSystemEncoding;
    }

    @Override
    public String content() {
      return contentForCharset(inputFile(), fileSystemEncoding);
    }

  }

  private static class Sq60File extends SonarQubePythonFile {

    public Sq60File(InputFile inputFile) {
      super(inputFile);
    }

    @Override
    public String content() {
      return contentForCharset(inputFile(), inputFile().charset());
    }

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
