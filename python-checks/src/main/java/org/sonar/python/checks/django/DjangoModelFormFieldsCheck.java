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
package org.sonar.python.checks.django;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.python.checks.django.DjangoUtils.getFieldAssignment;
import static org.sonar.python.checks.django.DjangoUtils.getMetaClass;

@Rule(key = "S6559")
public class DjangoModelFormFieldsCheck extends PythonSubscriptionCheck {

  public static final String ALL_MESSAGE = "Set the fields of this form explicitly instead of using \"__all__\".";
  public static final String EXCLUDE_MESSAGE = "Set the fields of this form explicitly instead of using \"exclude\".";

  private static final String DJANGO_MODEL_FORM_FQN = "django.forms.ModelForm";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      var classDef = (ClassDef) ctx.syntaxNode();
      if (TreeUtils.getParentClassesFQN(classDef).contains(DJANGO_MODEL_FORM_FQN)) {
        getMetaClass(classDef)
          .ifPresent(metaClass -> {
            getFieldAssignment(metaClass, "exclude")
              .ifPresent(exclude -> ctx.addIssue(exclude, EXCLUDE_MESSAGE));
            getFieldAssignment(metaClass, "fields")
              .filter(fields -> isAllAssignedValue(fields.assignedValue()))
              .ifPresent(fields -> ctx.addIssue(fields, ALL_MESSAGE));
          });
      }
    });
  }

  private static boolean isAllAssignedValue(Expression assignedValue) {
    return Optional.of(assignedValue)
      .filter(StringLiteral.class::isInstance)
      .map(StringLiteral.class::cast)
      .map(StringLiteral::trimmedQuotesValue)
      .filter("__all__"::equals)
      .isPresent();
  }
}
