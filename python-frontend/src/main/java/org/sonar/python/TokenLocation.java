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
package org.sonar.python;

import org.sonar.plugins.python.api.tree.Token;

public class TokenLocation {

  private final int startLine;
  private final int startLineOffset;
  private final int endLine;
  private final int endLineOffset;

  public TokenLocation(Token token) {
    this.startLine = token.line();
    this.startLineOffset = token.column();

    String value = token.value();
    String[] lines = value.split("\r\n|\n|\r", -1);

    if (lines.length > 1) {
      if (token.isCompressed()) {
        endLine = token.line();
        endLineOffset = this.startLineOffset + token.valueLength();
      } else {
        endLine = token.line() + lines.length - 1;
        endLineOffset = lines[lines.length - 1].length();
      }
    } else {
      this.endLine = this.startLine;
      this.endLineOffset = this.startLineOffset + token.valueLength();
    }
  }

  public int startLine() {
    return startLine;
  }

  public int startLineOffset() {
    return startLineOffset;
  }

  public int endLine() {
    return endLine;
  }

  public int endLineOffset() {
    return endLineOffset;
  }
}
