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

import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Trivia;

public class TriviaImpl implements Trivia {

  private final Token token;

  public TriviaImpl(Token triviaToken) {
    token = triviaToken;
  }

  @Override
  public Token token() {
    return token;
  }

  @Override
  public String value() {
    return token.value();
  }
}
