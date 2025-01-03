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

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Stream;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.v2.PythonType;

public class DictCompExpressionImpl extends PyTree implements DictCompExpression {

  private final Token openingBrace;
  private final Expression keyExpression;
  private final Token colon;
  private final Expression valueExpression;
  private final ComprehensionFor comprehensionFor;
  private final Token closingBrace;
  private Set<Symbol> symbols = new HashSet<>();
  private PythonType pythonType = PythonType.UNKNOWN;

  public DictCompExpressionImpl(Token openingBrace, Expression keyExpression, Token colon, Expression valueExpression,
                                ComprehensionFor compFor, Token closingBrace) {
    this.openingBrace = openingBrace;
    this.keyExpression = keyExpression;
    this.colon = colon;
    this.valueExpression = valueExpression;
    this.comprehensionFor = compFor;
    this.closingBrace = closingBrace;
  }

  @Override
  public Expression keyExpression() {
    return keyExpression;
  }

  @Override
  public Token colonToken() {
    return colon;
  }

  @Override
  public Expression valueExpression() {
    return valueExpression;
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
    visitor.visitDictCompExpression(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Stream.of(openingBrace, keyExpression, colon, valueExpression, comprehensionFor, closingBrace).filter(Objects::nonNull).toList();
  }

  @Override
  public Kind getKind() {
    return Kind.DICT_COMPREHENSION;
  }

  public void addLocalVariableSymbol(Symbol symbol) {
    symbols.add(symbol);
  }

  @Override
  public InferredType type() {
    return InferredTypes.DICT;
  }

  @Override
  public PythonType typeV2() {
    return this.pythonType;
  }

  public void typeV2(PythonType type) {
    this.pythonType = type;
  }
}
