/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.tree.LiteralPattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;


// TODO: should we consider None as NoneExpression?
public class LiteralPatternImpl extends PyTree implements LiteralPattern {

  private final Kind kind;
  private final List<Token> tokens;

  public LiteralPatternImpl(List<Token> tokens, Kind kind) {
    this.tokens = tokens;
    this.kind = kind;
  }

  @Override
  public String valueAsString() {
    return tokens.stream().map(Token::value).collect(Collectors.joining());
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitLiteralPattern(this);
  }

  @Override
  public Kind getKind() {
    return kind;
  }

  @Override
  List<Tree> computeChildren() {
    return Collections.unmodifiableList(tokens);
  }
}
