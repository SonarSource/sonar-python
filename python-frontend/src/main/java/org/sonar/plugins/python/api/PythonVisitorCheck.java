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
package org.sonar.plugins.python.api;

import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;

public abstract class PythonVisitorCheck extends BaseTreeVisitor implements PythonCheck {

  private PythonVisitorContext context;

  protected PythonVisitorContext getContext() {
    return context;
  }

  protected final PreciseIssue addIssue(Token token, @Nullable String message) {
    PreciseIssue newIssue = new PreciseIssue(this, IssueLocation.preciseLocation(token, message));
    getContext().addIssue(newIssue);
    return newIssue;
  }

  protected final PreciseIssue addIssue(Tree node, @Nullable String message) {
    PreciseIssue newIssue = new PreciseIssue(this, IssueLocation.preciseLocation(node, message));
    getContext().addIssue(newIssue);
    return newIssue;
  }

  @Override
  public void scanFile(PythonVisitorContext visitorContext) {
    this.context = visitorContext;
    scan(context.rootTree());
  }
}
