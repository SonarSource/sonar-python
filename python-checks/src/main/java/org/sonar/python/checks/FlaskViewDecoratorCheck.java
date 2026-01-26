/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S8374")
public class FlaskViewDecoratorCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Move this decorator to the \"decorators\" class attribute.";
  private static final String SECONDARY_MESSAGE = "This class inherits from a Flask View.";

  private static final String FLASK_VIEW_FQN = "flask.views.View";
  private static final TypeMatcher IS_FLASK_VIEW_MATCHER = TypeMatchers.isOrExtendsType(FLASK_VIEW_FQN);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, FlaskViewDecoratorCheck::checkClassDef);
  }

  private static void checkClassDef(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();

    if (classDef.decorators().isEmpty()) {
      return;
    }

    if (isFlaskViewClass(ctx, classDef)) {
      for (Decorator decorator : classDef.decorators()) {
        PreciseIssue issue = ctx.addIssue(decorator, MESSAGE);
        issue.secondary(classDef.name(), SECONDARY_MESSAGE);
      }
    }
  }

  private static boolean isFlaskViewClass(SubscriptionContext ctx, ClassDef classDef) {
    return IS_FLASK_VIEW_MATCHER.isTrueFor(classDef.name(), ctx);
  }
}

