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
import java.util.Map;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.python.IPythonLocation;


public class GeneratedIPythonFile implements PythonInputFile {

  InputFile wrappedFile;

  private String pythonContent;

  private Map<Integer, IPythonLocation> locationMap;

  public GeneratedIPythonFile(InputFile wrappedFile, String pythonContent, Map<Integer, IPythonLocation> locationMap) {
    this.locationMap = locationMap;
    this.wrappedFile = wrappedFile;
    this.pythonContent = pythonContent;
  }

  public Map<Integer, IPythonLocation> locationMap() {
    return locationMap;
  }

  @Override
  public InputFile wrappedFile() {
    return wrappedFile;
  }

  @Override
  public Kind kind() {
    return Kind.IPYTHON;
  }

  @Override
  public String toString() {
    return wrappedFile.toString();
  }

  @Override
  public String contents() throws IOException {
    return pythonContent;
  }

}
