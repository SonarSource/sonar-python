/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.checks;

import java.util.regex.Pattern;
import javax.annotation.CheckForNull;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.SubscriptionContext;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyFileInputTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyToken;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.api.tree.Tree.Kind;

@Rule(key = MissingDocstringCheck.CHECK_KEY)
public class MissingDocstringCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "S1720";

  private static final Pattern EMPTY_STRING_REGEXP = Pattern.compile("([bruBRU]+)?('\\s*')|(\"\\s*\")|('''\\s*''')|(\"\"\"\\s*\"\"\")");
  private static final String MESSAGE_NO_DOCSTRING = "Add a docstring to this %s.";
  private static final String MESSAGE_EMPTY_DOCSTRING = "The docstring for this %s should not be empty.";

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
    context.registerSyntaxNodeConsumer(Kind.FILE_INPUT, ctx -> checkDocString(ctx, ((PyFileInputTree) ctx.syntaxNode()).docstring()));
    context.registerSyntaxNodeConsumer(Kind.FUNCDEF, ctx -> checkDocString(ctx, ((PyFunctionDefTree) ctx.syntaxNode()).docstring()));
    context.registerSyntaxNodeConsumer(Kind.CLASSDEF, ctx -> checkDocString(ctx, ((PyClassDefTree) ctx.syntaxNode()).docstring()));
  }

  private static void checkDocString(SubscriptionContext ctx, @CheckForNull PyToken docstring) {
    Tree tree = ctx.syntaxNode();
    DeclarationType type = getType(tree);
    if (docstring == null) {
      raiseIssueNoDocstring(tree, type, ctx);
    } else if (EMPTY_STRING_REGEXP.matcher(docstring.value()).matches()) {
      raiseIssue(tree, MESSAGE_EMPTY_DOCSTRING, type, ctx);
    }
  }

  private static DeclarationType getType(Tree tree) {
    if (tree.is(Kind.FUNCDEF)) {
      if (((PyFunctionDefTree) tree).isMethodDefinition()) {
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
    if (type != DeclarationType.MODULE) {
      ctx.addIssue(getNameNode(tree), finalMessage);
    } else {
      ctx.addFileIssue(finalMessage);
    }
  }

  private static PyNameTree getNameNode(Tree tree) {
    if (tree.is(Kind.FUNCDEF)) {
      return ((PyFunctionDefTree) tree).name();
    }
    return ((PyClassDefTree) tree).name();
  }

}
