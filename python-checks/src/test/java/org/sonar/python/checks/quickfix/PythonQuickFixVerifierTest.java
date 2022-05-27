/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
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
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.python.checks.ClassMethodFirstArgumentNameCheck;
import org.sonar.python.quickfix.IssueWithQuickFix;
import org.sonar.python.quickfix.PythonQuickFix;
import org.sonar.python.quickfix.PythonTextEdit;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

public class PythonQuickFixVerifierTest {

  @Test
  public void test() {
    PythonCheck check = mock(PythonCheck.class);
    String codeWithIssue = "def vol():\n" +
      "    return length*width\n";
    String codeFixed = "def vol():\n" +
      "    return length*width*depth\n";

    PythonQuickFixVerifier.verifyNoQuickFix(check, codeWithIssue);
    LocationInFile loc = new LocationInFile(null, 2, 23, 2, 23);
    String replacement = "*depth";
    IssueLocation issueLocation = IssueLocation.preciseLocation(loc, replacement);
    IssueWithQuickFix issue = new IssueWithQuickFix(check, issueLocation);
    PythonQuickFix quickFix = PythonQuickFix.newQuickFix("Testing verifier")
      .addTextEdit(PythonTextEdit.insertAtPosition(issueLocation, replacement))
      .build();
    issue.addQuickFix(quickFix);

    String fixApply = PythonQuickFixVerifier.applyQuickFix(codeWithIssue, issue);
    assertThat(fixApply).isEqualTo(codeFixed);

    // For coverage purposes
    PythonSubscriptionCheck subscriptionCheck = mock(PythonSubscriptionCheck.class);
    PythonQuickFixVerifier.verifyNoQuickFix(subscriptionCheck, codeWithIssue);
  }

  @Test
  public void test_verify(){
    String codeWithIssue = "class A():\n" +
      "    @classmethod\n" +
      "    def area(bob): pass";
    String codeFixed = "class A():\n" +
      "    @classmethod\n" +
      "    def area(cls, bob): pass";
    PythonQuickFixVerifier.verify(new ClassMethodFirstArgumentNameCheck(), codeWithIssue, codeFixed);
  }
}
