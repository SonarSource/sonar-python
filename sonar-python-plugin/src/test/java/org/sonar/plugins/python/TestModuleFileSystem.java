package org.sonar.plugins.python;

import java.util.List;
import java.util.function.Consumer;
import org.sonar.api.batch.fs.InputFile;
import org.sonarsource.sonarlint.plugin.api.module.file.ModuleFileSystem;

public class TestModuleFileSystem implements ModuleFileSystem {

  private final List<InputFile> inputFiles;

  public TestModuleFileSystem(List<InputFile> inputFiles) {
    this.inputFiles = inputFiles;
  }

  @Override
  public void files(String s, InputFile.Type type, Consumer<InputFile> consumer) {
    inputFiles.forEach(consumer);
  }
}
