/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import org.sonar.plugins.python.api.tree.KeywordPattern;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class KeywordPatternImpl extends PyTree implements KeywordPattern {
  private final Name attributeName;
  private final Token equalToken;
  private final Pattern pattern;

  public KeywordPatternImpl(Name attributeName, Token equalToken, Pattern pattern) {
    this.attributeName = attributeName;
    this.equalToken = equalToken;
    this.pattern = pattern;
  }

  @Override
  public Name attributeName() {
    return attributeName;
  }

  @Override
  public Token equalToken() {
    return equalToken;
  }

  @Override
  public Pattern pattern() {
    return pattern;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitKeywordPattern(this);
  }

  @Override
  public Kind getKind() {
    return Kind.KEYWORD_PATTERN;
  }

  @Override
  List<Tree> computeChildren() {
    return Arrays.asList(attributeName, equalToken, pattern);
  }
}
