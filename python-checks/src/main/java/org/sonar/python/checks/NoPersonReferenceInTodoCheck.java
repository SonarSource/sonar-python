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

import com.intellij.psi.PsiElement;
import com.jetbrains.python.PyTokenTypes;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheck;

@Rule(key = "S1707")
public class NoPersonReferenceInTodoCheck extends PythonCheck {
  private static final String DEFAULT_PERSON_REFERENCE_PATTERN = "[ ]*\\([ _a-zA-Z0-9@.]+\\)";
  private static final String COMMENT_PATTERN = "^#[ ]*(todo|fixme)";

  @RuleProperty(
      key = "pattern",
      defaultValue = DEFAULT_PERSON_REFERENCE_PATTERN)
  public String personReferencePatternString = DEFAULT_PERSON_REFERENCE_PATTERN;

  @Override
  public void initialize(Context context) {
    final Pattern patternTodoFixme = Pattern.compile(COMMENT_PATTERN, Pattern.CASE_INSENSITIVE);
    final Pattern patternPersonReference = Pattern.compile(personReferencePatternString);
    context.registerSyntaxNodeConsumer(PyTokenTypes.END_OF_LINE_COMMENT, ctx -> {
      PsiElement node = ctx.syntaxNode();
      String comment = node.getText();
      Matcher matcher = patternTodoFixme.matcher(comment);
      if (matcher.find()) {
        String tail = comment.substring(matcher.end());
        if (!patternPersonReference.matcher(tail).find()) {
          ctx.addIssue(node, "Add a citation of the person who can best explain this comment.");
        }
      }
    });
  }

}

