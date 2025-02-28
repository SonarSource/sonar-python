/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
package org.sonar.plugins.python.architecture;

import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.api.scanner.ScannerSide;
import org.sonar.plugins.python.api.PythonFileConsumer;
import org.sonarsource.api.sonarlint.SonarLintSide;

/**
 * This wrapper will retrieve the ArchitectureCallback used to build Python UDGs from the scanner when SonarArchitecture is available
 * A dummy callback will be provided in environments where it is not available
 */
@ScannerSide
@SonarLintSide
public class ArchitectureCallbackWrapper {

  private final PythonFileConsumer architectureCallback;


  public ArchitectureCallbackWrapper() {
    this.architectureCallback = new DummyArchitectureCallback();
  }

  public ArchitectureCallbackWrapper(@Nullable PythonFileConsumer architectureCallback) {
    this.architectureCallback = Optional.ofNullable(architectureCallback).orElseGet(DummyArchitectureCallback::new);
  }

  public PythonFileConsumer architectureUdgBuilder() {
    return architectureCallback;
  }
}
