/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks.quickfix;

import org.junit.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.python.quickfix.TextEditUtils;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class PythonQuickFixVerifierTest {

  @Test
  public void no_issue_found() {
    PythonCheck check = new SimpleCheck();
    assertThatThrownBy(() -> PythonQuickFixVerifier.verify(check, "a,b,c", ""))
      .isInstanceOf(AssertionError.class)
      .hasMessage("[Number of issues] Expected 1 issue but found 0");
  }

  @Test
  public void more_than_one_issue_raised() {
    PythonCheck check = new SimpleCheck();
    assertThatThrownBy(() -> PythonQuickFixVerifier.verify(check, "a=10;b=3", ""))
      .isInstanceOf(AssertionError.class)
      .hasMessage("[Number of issues] Expected 1 issue but found 2");
  }

  @Test
  public void one_issue_raised_no_quickfix() {
    PythonCheck check = new SimpleCheckNoQuickFix();

    assertThatThrownBy(() -> PythonQuickFixVerifier.verify(check, "a=10", ""))
      .isInstanceOf(AssertionError.class)
      .hasMessage("[Number of quickfixes] Expected 1 quickfix but found 0");
  }

  @Test
  public void one_issue_one_qf_wrong_fix() {
    SimpleCheck simpleCheck = new SimpleCheck();
    assertThatThrownBy(() -> PythonQuickFixVerifier.verify(simpleCheck, "a=10", "a==10"))
      .isInstanceOf(AssertionError.class)
      .hasMessageContaining("[The code with the quickfix applied is not the expected result.\n" +
        "\"Applied QuickFixes are:\n" +
        "[a!=10]\n" +
        "Expected result:\n" +
        "[a==10]] expected:<[\"a[=]=10\"]> but was:<[\"a[!]=10\"]>");
  }

  @Test
  public void test_verify() {
    PythonQuickFixVerifier.verify(new SimpleCheck(), "a=10", "a!=10");
  }

  @Test
  public void verifyIPython() {
    PythonQuickFixVerifier.verifyIPython(new SimpleCheck(), "a=10\n?a", "a!=10\n?a");
    PythonQuickFixVerifier.verifyIPythonQuickFixMessages(new SimpleCheck(), "a=10\n?a", "Add '!' here.");
  }
  
  @Test
  public void test_multiple_lines() {
    PythonQuickFixVerifier.verify(new SimpleCheck(), "b \na=10", "b \na!=10");
  }

  private class SimpleCheck extends PythonSubscriptionCheck {

    @Override
    public void initialize(Context context) {
      context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, ctx -> {
        AssignmentStatement assignment = ((AssignmentStatement) ctx.syntaxNode());
        PreciseIssue issue = ctx.addIssue(assignment.equalTokens().get(0), "");

        PythonTextEdit text = TextEditUtils
          .insertBefore(assignment.equalTokens().get(0), "!");
        PythonQuickFix quickFix = PythonQuickFix.newQuickFix("Add '!' here.")
          .addTextEdit(text)
          .build();
        issue.addQuickFix(quickFix);
      });
    }
  }

  private class SimpleCheckNoQuickFix extends PythonSubscriptionCheck {
    @Override
    public void initialize(Context context) {
      context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, ctx -> ctx.addIssue(ctx.syntaxNode(), ""));
    }
  }
}
