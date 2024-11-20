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

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.PrintStatement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class PrintStatementImpl extends SimpleStatement implements PrintStatement {
  private final Token printKeyword;
  private final List<Expression> expressions;
  private final Separators separators;

  public PrintStatementImpl(Token printKeyword, List<Expression> expressions, Separators separators) {
    this.printKeyword = printKeyword;
    this.expressions = expressions;
    this.separators = separators;
  }

  @Override
  public Token printKeyword() {
    return printKeyword;
  }

  @Override
  public List<Expression> expressions() {
    return expressions;
  }

  @Override
  public Kind getKind() {
    return Kind.PRINT_STMT;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitPrintStatement(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(Collections.singletonList(printKeyword), expressions, separators.elements())
      .flatMap(List::stream).filter(Objects::nonNull).collect(Collectors.toList());
  }
}
