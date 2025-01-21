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
package org.sonar.plugins.python.warnings;

import javax.annotation.Nullable;
import org.sonar.api.notifications.AnalysisWarnings;
import org.sonar.api.scanner.ScannerSide;
import org.sonarsource.api.sonarlint.SonarLintSide;

/**
 * As {@link org.sonar.api.notifications.AnalysisWarnings} has been added in SQ 7.4, previous version of the API
 * do not have the class. Thus, in order to avoid a {@link ClassNotFoundException} at startup, we use a wrapper class
 * we know for sure will always be present. Depending on the sonar runtime, this wrapper will either forward the
 * warnings to the underlying {@link org.sonar.api.notifications.AnalysisWarnings} or do nothing when not available.
 */
@ScannerSide
@SonarLintSide(lifespan = SonarLintSide.MULTIPLE_ANALYSES)
public class AnalysisWarningsWrapper {
  private final AnalysisWarnings analysisWarnings;

  public AnalysisWarningsWrapper(@Nullable AnalysisWarnings analysisWarnings) {
    this.analysisWarnings = analysisWarnings;
  }

  public AnalysisWarningsWrapper() {
    this.analysisWarnings = null;
  }

  public void addUnique(String text) {
    if (analysisWarnings != null) {
      this.analysisWarnings.addUnique(text);
    }
  }
}
