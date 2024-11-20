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
import java.util.stream.Stream;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.SliceList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.HasTypeDependencies;
import org.sonar.python.types.InferredTypes;

import static org.sonar.python.types.InferredTypes.LIST;
import static org.sonar.python.types.InferredTypes.TUPLE;
import static org.sonar.python.types.InferredTypes.isDeclaredTypeWithTypeClass;

public class SliceExpressionImpl extends PyTree implements SliceExpression, HasTypeDependencies {

  private final Expression object;
  private final Token leftBracket;
  private final SliceList sliceList;
  private final Token rightBracket;

  public SliceExpressionImpl(Expression object, Token leftBracket, SliceList sliceList, Token rightBracket) {
    this.object = object;
    this.leftBracket = leftBracket;
    this.sliceList = sliceList;
    this.rightBracket = rightBracket;
  }

  @Override
  public Expression object() {
    return object;
  }

  @Override
  public Token leftBracket() {
    return leftBracket;
  }

  @Override
  public SliceList sliceList() {
    return sliceList;
  }

  @Override
  public Token rightBracket() {
    return rightBracket;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitSliceExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(object, leftBracket, sliceList, rightBracket).filter(Objects::nonNull).toList();
  }

  @Override
  public Kind getKind() {
    return Kind.SLICE_EXPR;
  }

  @Override
  public InferredType type() {
    InferredType objectType = object.type();
    if (objectType.equals(LIST)) {
      return LIST;
    }
    if (isDeclaredTypeWithTypeClass(objectType, BuiltinTypes.LIST)) {
      return objectType;
    }
    if (objectType.equals(TUPLE)) {
      return TUPLE;
    }
    if (isDeclaredTypeWithTypeClass(objectType, BuiltinTypes.TUPLE)) {
      return objectType;
    }
    return InferredTypes.anyType();
  }

  @Override
  public List<Expression> typeDependencies() {
    return Collections.singletonList(object);
  }
}
