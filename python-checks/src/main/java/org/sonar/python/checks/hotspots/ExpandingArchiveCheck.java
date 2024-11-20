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
package org.sonar.python.checks.hotspots;

import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.checks.AbstractCallExpressionCheck;

@Rule(key = "S5042")
public class ExpandingArchiveCheck extends AbstractCallExpressionCheck {

  @Override
  protected Set<String> functionsToCheck() {
    return immutableSet(
      "zipfile.ZipFile.extractall",
      "tarfile.open"
    );
  }

  @Override
  protected String message() {
    return "Make sure that expanding this archive file is safe here.";
  }
}
