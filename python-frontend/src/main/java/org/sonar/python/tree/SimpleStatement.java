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


import java.util.List;
import org.sonar.python.api.PythonPunctuator;
import org.sonar.python.api.PythonTokenType;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;

public abstract class SimpleStatement extends PyTree {

  //Returns the last child that is not a newline nor a semicolon
  @Override
  public Token lastToken() {
    if (lastToken == null) {
      List<Tree> children = children();
      Tree last = children.get(children.size() - 1);
      int index = 2;
      if (last.is(Kind.TOKEN) && ((Token) last).type() == PythonTokenType.NEWLINE) {
        last = children.get(children.size() - index);
        index++;
      }
      if (last.is(Kind.TOKEN) && ((Token) last).type() == PythonPunctuator.SEMICOLON) {
        last = children.get(children.size() - index);
      }
      this.lastToken = last.is(Kind.TOKEN) ? (Token) last : last.lastToken();
    }
    return lastToken;
  }
}
