/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.List;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DictionaryLiteralElement;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

public class DictionaryLiteralImpl extends DictOrSetLiteralImpl<DictionaryLiteralElement> implements DictionaryLiteral {

  public DictionaryLiteralImpl(Token lCurlyBrace, List<Token> commas, List<DictionaryLiteralElement> elements, Token rCurlyBrace) {
    super(lCurlyBrace, commas, elements, rCurlyBrace);
  }
  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitDictionaryLiteral(this);
  }

  @Override
  public Kind getKind() {
    return Kind.DICTIONARY_LITERAL;
  }

  @Override
  public InferredType type() {
    return InferredTypes.DICT;
  }
}
