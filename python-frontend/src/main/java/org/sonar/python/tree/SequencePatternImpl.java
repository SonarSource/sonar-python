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

import java.util.ArrayList;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.SequencePattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class SequencePatternImpl extends PyTree implements SequencePattern {

  @Nullable
  private final Token leftDelimiter; // might be '(' or '['
  private final List<Pattern> elements;
  private final List<Token> commas;
  @Nullable
  private final Token rightDelimiter; // might be ')' or ']'

  public SequencePatternImpl(@Nullable Token leftDelimiter, List<Pattern> elements, List<Token> commas, @Nullable Token rightDelimiter) {
    this.leftDelimiter = leftDelimiter;
    this.elements = elements;
    this.commas = commas;
    this.rightDelimiter = rightDelimiter;
  }

  @CheckForNull
  @Override
  public Token lDelimiter() {
    return leftDelimiter;
  }

  @Override
  public List<Pattern> elements() {
    return elements;
  }

  @Override
  public List<Token> commas() {
    return commas;
  }

  @CheckForNull
  @Override
  public Token rDelimiter() {
    return rightDelimiter;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitSequencePattern(this);
  }

  @Override
  public Kind getKind() {
    return Kind.SEQUENCE_PATTERN;
  }

  @Override
  List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>();
    if (leftDelimiter != null) {
      children.add(leftDelimiter);
    }
    int i = 0;
    for (Pattern element : elements) {
      children.add(element);
      if (i < commas.size()) {
        children.add(commas.get(i));
      }
      i++;
    }
    if (rightDelimiter != null) {
      children.add(rightDelimiter);
    }
    return children;
  }
}
