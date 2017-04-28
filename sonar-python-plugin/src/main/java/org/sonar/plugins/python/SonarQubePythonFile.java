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
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.python.PythonFile;

public class SonarQubePythonFile implements PythonFile {

  private final File file;
  private final Charset charset;

  public SonarQubePythonFile(InputFile inputFile, Charset charset) {
    this.file = inputFile.file();
    this.charset = charset;
  }

  @Override
  public String content() {
    try {
      return Files.toString(file, charset);
    } catch (IOException e) {
      throw new IllegalStateException("Cannot read " + file, e);
    }
  }

  @Override
  public String fileName() {
    return file.getName();
  }

}
