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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ExpressionListImpl extends PyTree implements ExpressionList {
  private final List<Expression> expressions;
  private final List<Token> commas;

  public ExpressionListImpl(List<Expression> expressions, List<Token> commas) {
    this.expressions = expressions;
    this.commas = commas;
  }

  @Override
  public List<Expression> expressions() {
    return expressions;
  }

  @Override
  public List<Token> commas() {
    return commas;
  }

  @Override
  public Kind getKind() {
    return Kind.EXPRESSION_LIST;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitExpressionList(this);
  }

  @Override
  public List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>();
    int i = 0;
    for (Expression expression : expressions) {
      children.add(expression);
      if (i < commas.size()) {
        children.add(commas.get(i));
      }
      i++;
    }
    return children;
  }
}
