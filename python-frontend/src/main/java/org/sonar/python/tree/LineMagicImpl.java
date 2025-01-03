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
import java.util.List;
import org.sonar.plugins.python.api.tree.LineMagic;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class LineMagicImpl extends PyTree implements LineMagic {

  private final Token percent;
  private final List<Token> tokens;

  public LineMagicImpl(Token percent, List<Token> tokens) {
    this.percent = percent;
    this.tokens = tokens;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    // no op
  }

  @Override
  public Kind getKind() {
    return Kind.LINE_MAGIC;
  }

  @Override
  List<Tree> computeChildren() {
    var children = new ArrayList<Tree>();
    children.add(percent);
    children.addAll(tokens);
    return children;
  }

}
