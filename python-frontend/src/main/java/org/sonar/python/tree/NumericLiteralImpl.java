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

import java.util.Collections;
import java.util.List;
import java.util.Locale;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;

public class NumericLiteralImpl extends PyTree implements NumericLiteral {

  private final String valueAsString;
  private final Token token;
  private final InferredType type;

  NumericLiteralImpl(Token token) {
    this.token = token;
    valueAsString = token.value();
    type = computeType();
  }

  @Override
  public Kind getKind() {
    return Kind.NUMERIC_LITERAL;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitNumericLiteral(this);
  }

  @Override
  public long valueAsLong() {
    String literalValue = valueAsString.replace("_", "");
    if (literalValue.startsWith("0b") || literalValue.startsWith("0B")) {
      return Integer.valueOf(literalValue.substring(2), 2);
    }
    if (literalValue.startsWith("0o") || literalValue.startsWith("0O")) {
      return Integer.valueOf(literalValue.substring(2), 8);
    } else if (literalValue.startsWith("0") && literalValue.length() > 1) {
      // Python 2 syntax (https://www.python.org/dev/peps/pep-3127/#removal-of-old-octal-syntax)
      return Integer.valueOf(literalValue.substring(1), 8);
    }
    if (valueAsString.endsWith("L") || valueAsString.endsWith("l")) {
      literalValue = literalValue.substring(0, literalValue.length() - 1);
    }
    return Long.parseLong(literalValue);
  }

  @Override
  public String valueAsString() {
    return valueAsString;
  }

  @Override
  public List<Tree> computeChildren() {
    return Collections.singletonList(token);
  }

  @Override
  public InferredType type() {
    return type;
  }

  // https://docs.python.org/3/reference/lexical_analysis.html#numeric-literals
  private InferredType computeType() {
    String valueAsStringLowerCase = valueAsString.toLowerCase(Locale.ROOT);
    if (valueAsStringLowerCase.contains("j")) {
      return InferredTypes.COMPLEX;
    }
    if (valueAsString.contains(".") || valueAsStringLowerCase.contains("e")) {
      return InferredTypes.FLOAT;
    }
    return InferredTypes.INT;
  }
}
