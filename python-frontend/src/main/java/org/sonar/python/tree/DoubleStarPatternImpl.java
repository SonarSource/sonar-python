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

import java.util.Arrays;
import java.util.List;
import org.sonar.plugins.python.api.tree.CapturePattern;
import org.sonar.plugins.python.api.tree.DoubleStarPattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class DoubleStarPatternImpl extends PyTree implements DoubleStarPattern {

  private final Token doubleStarToken;
  private final CapturePattern capturePattern;

  public DoubleStarPatternImpl(Token doubleStarToken, CapturePattern capturePattern) {
    this.doubleStarToken = doubleStarToken;
    this.capturePattern = capturePattern;
  }
  @Override
  public Token doubleStarToken() {
    return doubleStarToken;
  }

  @Override
  public CapturePattern capturePattern() {
    return capturePattern;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitDoubleStarPattern(this);
  }

  @Override
  public Kind getKind() {
    return Kind.DOUBLE_STAR_PATTERN;
  }

  @Override
  List<Tree> computeChildren() {
    return Arrays.asList(doubleStarToken, capturePattern);
  }
}
