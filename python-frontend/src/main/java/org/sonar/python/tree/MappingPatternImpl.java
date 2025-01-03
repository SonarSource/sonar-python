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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.sonar.plugins.python.api.tree.MappingPattern;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class MappingPatternImpl extends PyTree implements MappingPattern {

  private final Token lCurlyBrace;
  private final List<Token> commas;
  private final List<Pattern> elements;
  private final Token rCurlyBrace;

  public MappingPatternImpl(Token lCurlyBrace, List<Token> commas, List<Pattern> elements, Token rCurlyBrace) {
    this.lCurlyBrace = lCurlyBrace;
    this.commas = commas;
    this.elements = elements;
    this.rCurlyBrace = rCurlyBrace;
  }

  @Override
  public Token lCurlyBrace() {
    return lCurlyBrace;
  }

  @Override
  public List<Pattern> elements() {
    return Collections.unmodifiableList(elements);
  }

  @Override
  public List<Token> commas() {
    return Collections.unmodifiableList(commas);
  }

  @Override
  public Token rCurlyBrace() {
    return rCurlyBrace;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitMappingPattern(this);
  }

  @Override
  public Kind getKind() {
    return Kind.MAPPING_PATTERN;
  }

  @Override
  public List<Tree> computeChildren() {
    List<Tree> child = new ArrayList<>();
    child.add(lCurlyBrace);
    int i = 0;
    for (Pattern element : elements) {
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
