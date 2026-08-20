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
package org.sonar.plugins.python.indexer;

import java.util.List;
import java.util.stream.Stream;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.plugins.python.PythonInputFile;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileSystem;

public class TestModuleFileSystem implements ModuleFileSystem {

  private final List<PythonInputFile> inputFiles;

  public TestModuleFileSystem(List<PythonInputFile> inputFiles) {
    this.inputFiles = inputFiles;
  }

  @Override
  public Stream<InputFile> files(String language, InputFile.Type type) {
    return files()
      .filter(file -> language.equals(file.language()))
      .filter(file -> type == file.type());
  }

  @Override
  public Stream<InputFile> files() {
    return inputFiles.stream().map(PythonInputFile::wrappedFile);
  }

  public void addFile(PythonInputFile inputFile) {
    inputFiles.add(inputFile);
  }
}
