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
import org.sonar.plugins.python.api.tree.GroupPattern;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class GroupPatternImpl extends PyTree implements GroupPattern {

  private final Token leftPar;
  private final Pattern pattern;
  private final Token rightPar;

  public GroupPatternImpl(Token leftPar, Pattern pattern, Token rightPar) {
    this.leftPar = leftPar;
    this.pattern = pattern;
    this.rightPar = rightPar;
  }

  @Override
  public Token leftPar() {
    return leftPar;
  }

  @Override
  public Pattern pattern() {
    return pattern;
  }

  @Override
  public Token rightPar() {
    return rightPar;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitGroupPattern(this);
  }

  @Override
  public Kind getKind() {
    return Kind.GROUP_PATTERN;
  }

  @Override
  List<Tree> computeChildren() {
    return Arrays.asList(leftPar, pattern, rightPar);
  }
}
