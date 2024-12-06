/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonLine;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.TextEditUtils;

@Rule(key = "S1720")
public class MissingDocstringCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE_NO_DOCSTRING = "Add a docstring to this %s.";
  private static final String MESSAGE_EMPTY_DOCSTRING = "The docstring for this %s should not be empty.";

  private static final String EMPTY_DOCSTRING = "\"\"\" doc \"\"\"";

  private enum DeclarationType {
    MODULE("module"),
    CLASS("class"),
    METHOD("method"),
    FUNCTION("function");

    private final String value;

    DeclarationType(String value) {
      this.value = value;
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, ctx -> checkFileInput(ctx, (FileInput) ctx.syntaxNode()));
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> checkDocString(ctx, ((FunctionDef) ctx.syntaxNode()).docstring()));
    context.registerSyntaxNodeConsumer(Kind.CLASSDEF, ctx -> checkDocString(ctx, ((ClassDef) ctx.syntaxNode()).docstring()));
  }

  private static void checkFileInput(SubscriptionContext ctx, FileInput fileInput) {
    if ("__init__.py".equals(ctx.pythonFile().fileName()) && fileInput.statements() == null) {
      return;
    }
    checkDocString(ctx, fileInput.docstring());
  }

  private static void checkDocString(SubscriptionContext ctx, @CheckForNull StringLiteral docstring) {
    Tree tree = ctx.syntaxNode();
    DeclarationType type = getType(tree);
    if (docstring == null) {
      raiseIssueNoDocstring(tree, type, ctx);
    } else if (docstring.trimmedQuotesValue().trim().length() == 0) {
      raiseIssue(tree, MESSAGE_EMPTY_DOCSTRING, type, ctx);
    }
  }

  private static DeclarationType getType(Tree tree) {
    if (tree.is(Kind.FUNCDEF)) {
      if (((FunctionDef) tree).isMethodDefinition()) {
        return DeclarationType.METHOD;
      } else {
        return DeclarationType.FUNCTION;
      }
    } else if (tree.is(Kind.CLASSDEF)) {
      return DeclarationType.CLASS;
    } else {
      // tree is FILE_INPUT
      return DeclarationType.MODULE;
    }
  }

  private static void raiseIssueNoDocstring(Tree tree, DeclarationType type, SubscriptionContext ctx) {
    if (type != DeclarationType.METHOD) {
      raiseIssue(tree, MESSAGE_NO_DOCSTRING, type, ctx);
    }
  }

  private static void raiseIssue(Tree tree, String message, DeclarationType type, SubscriptionContext ctx) {
    String finalMessage = String.format(message, type.value);
    PreciseIssue issue;
    if (type != DeclarationType.MODULE) {
      issue = ctx.addIssue(getNameNode(tree), finalMessage);
    } else {
      issue = ctx.addFileIssue(finalMessage);
    }
    addQuickFix(issue, tree, type);
  }

  private static Name getNameNode(Tree tree) {
    if (tree.is(Kind.FUNCDEF)) {
      return ((FunctionDef) tree).name();
    }
    return ((ClassDef) tree).name();
  }

  private static void addQuickFix(PreciseIssue issue, Tree tree, DeclarationType type) {
    PythonQuickFix.Builder quickFix = PythonQuickFix.newQuickFix("Add docstring");

    if (type == DeclarationType.MODULE) {
      quickFix.addTextEdit(TextEditUtils.insertAtPosition(new PythonLine(1), 0, EMPTY_DOCSTRING));
    } else if (type == DeclarationType.CLASS) {
      ClassDef classDef = (ClassDef) tree;
      quickFix.addTextEdit(TextEditUtils.insertLineAfter(classDef.colon(), classDef.body(), EMPTY_DOCSTRING));
    } else {
      FunctionDef functionDef = (FunctionDef) tree;
      quickFix.addTextEdit(TextEditUtils.insertLineAfter(functionDef.colon(), functionDef.body(), EMPTY_DOCSTRING));
    }

    issue.addQuickFix(quickFix.build());
  }

}
