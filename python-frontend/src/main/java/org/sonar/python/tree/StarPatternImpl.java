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

import java.util.Arrays;
import java.util.List;
import org.sonar.plugins.python.api.tree.CapturePattern;
import org.sonar.plugins.python.api.tree.StarPattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class StarPatternImpl extends PyTree implements StarPattern {

  private final Token starToken;
  private final CapturePattern capturePattern;

  public StarPatternImpl(Token starToken, CapturePattern capturePattern) {
    this.starToken = starToken;
    this.capturePattern = capturePattern;
  }

  @Override
  public CapturePattern capturePattern() {
    return capturePattern;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitStarPattern(this);
  }

  @Override
  public Kind getKind() {
    return Kind.STAR_PATTERN;
  }

  @Override
  List<Tree> computeChildren() {
    return Arrays.asList(starToken, capturePattern);
  }
}
