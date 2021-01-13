/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python.tree;

import java.util.ArrayList;
import java.util.List;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;

public abstract class DictOrSetLiteralImpl<E extends Tree> extends PyTree {
  private final Token lCurlyBrace;
  private final List<Token> commas;
  private final List<E> elements;
  private final Token rCurlyBrace;

  public DictOrSetLiteralImpl(Token lCurlyBrace, List<Token> commas, List<E> elements, Token rCurlyBrace) {
    this.lCurlyBrace = lCurlyBrace;
    this.commas = commas;
    this.elements = elements;
    this.rCurlyBrace = rCurlyBrace;
  }

  public Token lCurlyBrace() {
    return lCurlyBrace;
  }

  public Token rCurlyBrace() {
    return rCurlyBrace;
  }

  public List<Token> commas() {
    return commas;
  }

  public List<E> elements() {
    return elements;
  }

  @Override
  public List<Tree> computeChildren() {
    List<Tree> child = new ArrayList<>();
    child.add(lCurlyBrace);
    int i = 0;
    for (E element : elements) {
      child.add(element);
      if (i < commas.size()) {
        child.add(commas.get(i));
      }
      i++;
    }
    child.add(rCurlyBrace);
    return child;
  }
}
