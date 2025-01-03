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
import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.KeyValuePair;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class KeyValuePairImpl extends PyTree implements KeyValuePair {
  private final Expression key;
  private final Token colon;
  private final Expression value;

  public KeyValuePairImpl(Expression key, Token colon, Expression value) {
    this.key = key;
    this.colon = colon;
    this.value = value;
  }

  @CheckForNull
  @Override
  public Expression key() {
    return key;
  }

  @CheckForNull
  @Override
  public Token colon() {
    return colon;
  }

  @CheckForNull
  @Override
  public Expression value() {
    return value;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitKeyValuePair(this);
  }

  @Override
  public List<Tree> computeChildren() {
    return Arrays.asList(key, colon, value);
  }

  @Override
  public Kind getKind() {
    return Kind.KEY_VALUE_PAIR;
  }
}
