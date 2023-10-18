/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.lexer;

import java.util.ArrayDeque;
import java.util.Deque;

import org.sonar.python.lexer.FStringState.Mode;

public class LexerState {

  public final Deque<Integer> indentationStack = new ArrayDeque<>();

  public final Deque<FStringState> fStringStateStack = new ArrayDeque<>();

  int brackets;
  boolean joined;
  int initialLine = 1;
  int initialColumn = 0;

  public void reset() {
    indentationStack.clear();
    indentationStack.push(0);

    brackets = 0;
    joined = false;
    fStringStateStack.clear();
    fStringStateStack.push(new FStringState(Mode.REGULAR_MODE, brackets));
  }

  public void reset(int initialLine, int initialColumn) {
    reset();
    this.initialLine = initialLine;
    this.initialColumn = initialColumn;
  }

}
