/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.v2.PythonType;

public class ComprehensionExpressionImpl extends PyTree implements ComprehensionExpression {

  private final Kind kind;
  private final Token openingToken;
  private final Expression resultExpression;
  private final ComprehensionFor comprehensionFor;
  private final Token closingToken;
  private Set<Symbol> symbols = new HashSet<>();
  private PythonType typeV2;

  public ComprehensionExpressionImpl(Kind kind, @Nullable Token openingToken, Expression resultExpression,
                                     ComprehensionFor compFor, @Nullable Token closingToken) {
    this.kind = kind;
    this.resultExpression = resultExpression;
    this.comprehensionFor = compFor;
    this.openingToken = openingToken;
    this.closingToken = closingToken;
  }

  @Override
  public Expression resultExpression() {
    return resultExpression;
  }

  @Override
  public ComprehensionFor comprehensionFor() {
    return comprehensionFor;
  }

  @Override
  public Set<Symbol> localVariables() {
    return symbols;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitPyListOrSetCompExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(openingToken, resultExpression, comprehensionFor, closingToken).filter(Objects::nonNull).toList();
  }

  @Override
  public Kind getKind() {
    return kind;
  }

  public void addLocalVariableSymbol(Symbol symbol) {
    symbols.add(symbol);
  }

  @Override
  public InferredType type() {
    switch (kind) {
      case LIST_COMPREHENSION:
        return InferredTypes.LIST;
      case SET_COMPREHENSION:
        return InferredTypes.SET;
      default:
        // GENERATOR_EXPR: needs a class symbol for 'generator'
        return InferredTypes.anyType();
    }
  }

  @Override
  public PythonType typeV2() {
    return this.typeV2;
  }

  public void typeV2(PythonType type) {
    this.typeV2 = type;
  }
}
