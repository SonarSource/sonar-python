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
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.StarPattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class StarPatternImpl extends PyTree implements StarPattern {

  private final Token starToken;
  private final Pattern pattern;

  public StarPatternImpl(Token starToken, Pattern pattern) {
    this.starToken = starToken;
    this.pattern = pattern;
  }

  @Override
  public Token starToken() {
    return starToken;
  }

  @Override
  public Pattern pattern() {
    return pattern;
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
    return Arrays.asList(starToken, pattern);
  }
}
