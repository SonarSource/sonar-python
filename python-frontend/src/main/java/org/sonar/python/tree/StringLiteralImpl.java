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

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.v2.PythonType;

public class StringLiteralImpl extends PyTree implements StringLiteral {

  private final List<StringElement> stringElements;
  private static final Set<String> BYTES_PREFIXES = new HashSet<>(Arrays.asList("b", "B", "br", "Br", "bR", "BR", "rb", "rB", "Rb", "RB"));
  private PythonType typeV2 = PythonType.UNKNOWN;

  StringLiteralImpl(List<StringElement> stringElements) {
    this.stringElements = stringElements;
  }

  @Override
  public Kind getKind() {
    return Kind.STRING_LITERAL;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitStringLiteral(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Collections.unmodifiableList(stringElements);
  }

  @Override
  public List<StringElement> stringElements() {
    return stringElements;
  }

  @Override
  public String trimmedQuotesValue() {
    return stringElements().stream()
      .map(StringElement::trimmedQuotesValue)
      .collect(Collectors.joining());
  }

  // https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals
  @Override
  public InferredType type() {
    if (stringElements.size() == 1 && BYTES_PREFIXES.contains(stringElements.get(0).prefix())) {
      // Python 3: bytes, Python 2: str
      return InferredTypes.anyType();
    }
    return InferredTypes.STR;
  }

  @Override
  public PythonType typeV2() {
    return this.typeV2;
  }

  public void typeV2(PythonType type) {
    this.typeV2 = type;
  }
}
