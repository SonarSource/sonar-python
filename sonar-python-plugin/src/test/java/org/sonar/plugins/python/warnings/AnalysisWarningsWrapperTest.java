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
package org.sonar.plugins.python.warnings;

import org.junit.jupiter.api.Test;
import org.sonar.api.notifications.AnalysisWarnings;

import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class AnalysisWarningsWrapperTest {

  @Test
  void test() {
    AnalysisWarnings analysisWarnings = spy(AnalysisWarnings.class);
    AnalysisWarningsWrapper defaultAnalysisWarningsWrapper = new AnalysisWarningsWrapper(analysisWarnings);
    defaultAnalysisWarningsWrapper.addUnique("abcd");
    defaultAnalysisWarningsWrapper.addUnique("def");
    verify(analysisWarnings, times(2)).addUnique(anyString());
  }

}
