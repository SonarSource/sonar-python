/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ComprehensionFor;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.tree.UnpackingExpression;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;
import org.sonar.plugins.python.api.types.v2.PythonType;

public class DictCompExpressionImpl extends PyTree implements DictCompExpression {

  private final Token openingBrace;
  private final DictionaryLiteralElement resultElement;
  private final ComprehensionFor comprehensionFor;
  private final Token closingBrace;
  private final Set<Symbol> symbols = new HashSet<>();
  private PythonType pythonType = PythonType.UNKNOWN;

  public DictCompExpressionImpl(Token openingBrace, Expression keyExpression, Token colon, Expression valueExpression,
                                ComprehensionFor compFor, Token closingBrace) {
    this(openingBrace, new KeyValuePairImpl(keyExpression, colon, valueExpression), compFor, closingBrace);
  }

  public DictCompExpressionImpl(Token openingBrace, UnpackingExpression unpackingExpression, ComprehensionFor compFor, Token closingBrace) {
    this(openingBrace, (DictionaryLiteralElement) unpackingExpression, compFor, closingBrace);
  }

  public DictCompExpressionImpl(Token openingBrace, DictionaryLiteralElement resultElement, ComprehensionFor compFor, Token closingBrace) {
    this.openingBrace = openingBrace;
    if (!(resultElement instanceof KeyValuePair)
      && !(resultElement instanceof UnpackingExpression unpacking && "**".equals(unpacking.starToken().value()))) {
      throw new IllegalArgumentException("A dictionary comprehension head must be a key/value pair or a mapping unpacking expression");
    }
    this.resultElement = resultElement;
    this.comprehensionFor = compFor;
    this.closingBrace = closingBrace;
  }

  @Override
  public DictionaryLiteralElement resultElement() {
    return resultElement;
  }

  /**
   * @deprecated Use {@link #resultElement()} and {@link KeyValuePair#key()}.
   */
  @Override
  @Deprecated(since = "5.33")
  @CheckForNull
  public Expression keyExpression() {
    return resultElement instanceof KeyValuePair pair ? pair.key() : null;
  }

  /**
   * @deprecated Use {@link #resultElement()} and {@link KeyValuePair#colon()}.
   */
  @Override
  @Deprecated(since = "5.33")
  @CheckForNull
  public Token colonToken() {
    return resultElement instanceof KeyValuePair pair ? pair.colon() : null;
  }

  /**
   * @deprecated Use {@link #resultElement()} and {@link KeyValuePair#value()}.
   */
  @Override
  @Deprecated(since = "5.33")
  @CheckForNull
  public Expression valueExpression() {
    return resultElement instanceof KeyValuePair pair ? pair.value() : null;
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
    return List.of(openingBrace, resultElement, comprehensionFor, closingBrace);
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
