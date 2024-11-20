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

import java.util.List;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.types.InferredTypes;

public class QualifiedExpressionImpl extends PyTree implements QualifiedExpression {
  private final Name name;
  private final Expression qualifier;
  private final Token dotToken;

  public QualifiedExpressionImpl(Name name, Expression qualifier, Token dotToken) {
    this.name = name;
    this.qualifier = qualifier;
    this.dotToken = dotToken;
  }

  @Override
  public Expression qualifier() {
    return qualifier;
  }

  @Override
  public Token dotToken() {
    return dotToken;
  }

  @Override
  public Name name() {
    return name;
  }

  @Override
  public Kind getKind() {
    return Kind.QUALIFIED_EXPR;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitQualifiedExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(qualifier, dotToken, name).toList();
  }

  @Override
  public InferredType type() {
    Symbol symbol = name.symbol();
    if (symbol == null) return InferredTypes.anyType();
    return ((SymbolImpl) symbol).inferredType();
  }
}
