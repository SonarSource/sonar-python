/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.tree;

import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.FunctionSymbolImpl;
import org.sonar.python.types.InferredTypes;

public class CallExpressionImpl extends PyTree implements CallExpression {
  private final Expression callee;
  private final ArgList argumentList;
  private final Token leftPar;
  private final Token rightPar;

  public CallExpressionImpl(Expression callee, @Nullable ArgList argumentList, Token leftPar, Token rightPar) {
    this.callee = callee;
    this.argumentList = argumentList;
    this.leftPar = leftPar;
    this.rightPar = rightPar;
  }

  @Override
  public Expression callee() {
    return callee;
  }

  @Override
  public ArgList argumentList() {
    return argumentList;
  }

  @Override
  public List<Argument> arguments() {
    return argumentList != null ? argumentList.arguments() : Collections.emptyList();
  }

  @Override
  public Token leftPar() {
    return leftPar;
  }

  @Override
  public Token rightPar() {
    return rightPar;
  }

  @Override
  public Kind getKind() {
    return Kind.CALL_EXPR;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitCallExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(callee, leftPar, argumentList, rightPar).filter(Objects::nonNull).collect(Collectors.toList());
  }

  @Override
  public InferredType type() {
    Symbol calleeSymbol = calleeSymbol();
    if (calleeSymbol != null) {
      if (calleeSymbol.kind() == Symbol.Kind.CLASS) {
        ClassSymbol classSymbol = (ClassSymbol) calleeSymbol;
        return InferredTypes.runtimeType(classSymbol.fullyQualifiedName());
      }
      if (calleeSymbol.kind() == Symbol.Kind.FUNCTION) {
        FunctionSymbolImpl functionSymbol = (FunctionSymbolImpl) calleeSymbol;
        return functionSymbol.inferredReturnType();
      }
    }
    return InferredTypes.anyType();
  }
}
