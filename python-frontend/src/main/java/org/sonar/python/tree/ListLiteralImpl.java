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

import java.util.List;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.v2.PythonType;

public class ListLiteralImpl extends PyTree implements ListLiteral {

  private final Token leftBracket;
  private final ExpressionList elements;
  private final Token rightBracket;

  private PythonType typeV2 = PythonType.UNKNOWN;

  public ListLiteralImpl(Token leftBracket, ExpressionList elements, Token rightBracket) {
    this.leftBracket = leftBracket;
    this.elements = elements;
    this.rightBracket = rightBracket;
  }

  @Override
  public Kind getKind() {
    return Kind.LIST_LITERAL;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitListLiteral(this);
  }

  @Override
  public Token leftBracket() {
    return leftBracket;
  }

  @Override
  public ExpressionList elements() {
    return elements;
  }

  @Override
  public Token rightBracket() {
    return rightBracket;
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(leftBracket, elements, rightBracket).toList();
  }

  @Override
  public InferredType type() {
    return InferredTypes.LIST;
  }

  @Override
  public PythonType typeV2() {
    return this.typeV2;
  }

  public void typeV2(PythonType type) {
    this.typeV2 = type;
  }
}
