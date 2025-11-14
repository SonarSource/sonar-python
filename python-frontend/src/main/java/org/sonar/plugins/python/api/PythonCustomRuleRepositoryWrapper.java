/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.api;

import javax.annotation.Nullable;

import org.sonar.api.ExtensionPoint;
import org.sonar.api.scanner.ScannerSide;
import org.sonarsource.api.sonarlint.SonarLintSide;

@ScannerSide
@SonarLintSide
@ExtensionPoint
public class PythonCustomRuleRepositoryWrapper {

  private final PythonCustomRuleRepository[] customRuleRepositories;

  public PythonCustomRuleRepositoryWrapper() {
    // Constructor to be used by pico if no customRuleRepositories are to be found and injected.
    this(new PythonCustomRuleRepository[0]);
  }

  public PythonCustomRuleRepositoryWrapper(@Nullable PythonCustomRuleRepository[] customRuleRepositories) {
    if(customRuleRepositories == null) {
      customRuleRepositories = new PythonCustomRuleRepository[0];
    }
    this.customRuleRepositories = customRuleRepositories;
  }

  public PythonCustomRuleRepository[] customRuleRepositories() {
    return customRuleRepositories;
  }

}
