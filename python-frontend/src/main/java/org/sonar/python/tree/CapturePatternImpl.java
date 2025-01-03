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

import java.util.Collections;
import java.util.List;
import org.sonar.plugins.python.api.tree.CapturePattern;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TreeVisitor;

public class CapturePatternImpl extends PyTree implements CapturePattern {

  private final Name name;

  public CapturePatternImpl(Name name) {
    this.name = name;
  }

  @Override
  public Name name() {
    return name;
  }

  @Override
  public void accept(TreeVisitor visitor) {
    visitor.visitCapturePattern(this);
  }

  @Override
  public Kind getKind() {
    return Kind.CAPTURE_PATTERN;
  }

  @Override
  List<Tree> computeChildren() {
    return Collections.singletonList(name);
  }
}
