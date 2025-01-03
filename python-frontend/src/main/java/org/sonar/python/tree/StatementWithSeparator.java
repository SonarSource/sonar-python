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
package org.sonar.python.tree;

import com.sonar.sslr.api.AstNode;
import javax.annotation.Nullable;

public class StatementWithSeparator {
  private AstNode statement;
  private Separators separators;

  public StatementWithSeparator(AstNode statement, @Nullable Separators separators) {
    this.statement = statement;
    this.separators = separators == null ? Separators.EMPTY : separators;
  }

  public AstNode statement() {
    return statement;
  }

  public Separators separator() {
    return separators;
  }
}
