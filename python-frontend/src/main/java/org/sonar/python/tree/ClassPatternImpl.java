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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.sonar.plugins.python.api.tree.ClassPattern;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Pattern;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class ClassPatternImpl extends PyTree implements ClassPattern {

  private final Expression targetClass;
  private final Token leftPar;
  private final List<Pattern> arguments;
  private final List<Token> argumentSeparators;
  private final Token rightPar;

  public ClassPatternImpl(Expression targetClass, Token leftPar, List<Pattern> arguments, List<Token> argumentSeparators, Token rightPar) {
    this.targetClass = targetClass;
    this.leftPar = leftPar;
    this.arguments = arguments;
    this.argumentSeparators = argumentSeparators;
    this.rightPar = rightPar;
  }

  @Override
  public Expression targetClass() {
    return targetClass;
  }

  @Override
  public Token leftPar() {
    return leftPar;
  }

  @Override
  public List<Pattern> arguments() {
    return arguments;
  }

  @Override
  public List<Token> argumentSeparators() {
    return argumentSeparators;
  }

  @Override
  public Token rightPar() {
    return rightPar;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitClassPattern(this);
  }

  @Override
  public Kind getKind() {
    return Kind.CLASS_PATTERN;
  }

  @Override
  List<Tree> computeChildren() {
    List<Tree> children = new ArrayList<>(Arrays.asList(targetClass, leftPar));
    int i = 0;
    for (Pattern element : arguments) {
      children.add(element);
      if (i < argumentSeparators.size()) {
        children.add(argumentSeparators.get(i));
      }
      i++;
    }
    children.add(rightPar);
    return children;
  }
}
