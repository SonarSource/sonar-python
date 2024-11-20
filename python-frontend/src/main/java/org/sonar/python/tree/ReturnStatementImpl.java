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
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ReturnStatementImpl extends SimpleStatement implements ReturnStatement {
  private final Token returnKeyword;
  private final List<Expression> expressionTrees;
  private final List<Token> commas;
  private final Separators separators;

  public ReturnStatementImpl(Token returnKeyword, List<Expression> expressionTrees, List<Token> commas, Separators separators) {
    this.returnKeyword = returnKeyword;
    this.expressionTrees = expressionTrees;
    this.commas = commas;
    this.separators = separators;
  }

  @Override
  public Token returnKeyword() {
    return returnKeyword;
  }

  @Override
  public List<Expression> expressions() {
    return expressionTrees;
  }

  @Override
  public List<Token> commas() {
    return commas;
  }

  @Override
  public Kind getKind() {
    return Kind.RETURN_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitReturnStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>();
    children.add(returnKeyword);
    int i = 0;
    for (Expression expression : expressionTrees) {
      children.add(expression);
      if (i < commas.size()) {
        children.add(commas.get(i));
      }
      i++;
    }
    children.addAll(separators.elements());
    return children;
  }

  @Override
  public Token separator() {
    return separators.last();
  }
}
