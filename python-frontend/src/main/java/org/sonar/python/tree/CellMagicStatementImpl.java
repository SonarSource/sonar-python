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
import org.sonar.plugins.python.api.tree.CellMagicStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;


public class CellMagicStatementImpl extends PyTree implements CellMagicStatement {

  private final List<Token> tokens;

  public CellMagicStatementImpl(List<Token> tokens) {
    this.tokens = tokens;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    // no op
  }

  @Override
  public Kind getKind() {
    return Kind.CELL_MAGIC_STATEMENT;
  }

  @Override
  List<Tree> computeChildren() {
    return new ArrayList<>(tokens);
  }
}
