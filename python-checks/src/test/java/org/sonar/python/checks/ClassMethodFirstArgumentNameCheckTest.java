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
package org.sonar.python.checks;

import java.util.List;
import org.junit.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.python.checks.utils.PythonCheckVerifier;
import org.sonar.python.checks.utils.PythonQuickFixVerifier;
import org.sonar.python.quickfix.IssueWithQuickFix;

import static org.assertj.core.api.Assertions.assertThat;

public class ClassMethodFirstArgumentNameCheckTest {

  @Test
  public void test() {
    PythonCheckVerifier.verify("src/test/resources/checks/classMethodFirstArgumentNameCheck.py", new ClassMethodFirstArgumentNameCheck());

    String codeWithIssue = "class A():\n\n" +
      "    @classmethod\n" +
      "    def area(bob, length, width):\n" +
      "        return length*width\n";
    String codeFixed = "class A():\n\n" +
      "    @classmethod\n" +
      "    def area(cls, bob, length, width):\n" +
      "        return length*width\n";

    List<PythonCheck.PreciseIssue> issues = PythonQuickFixVerifier
      .getIssuesWithQuickFix(codeWithIssue, new ClassMethodFirstArgumentNameCheck());

    assertThat(issues).hasSize(1);
    IssueWithQuickFix issue = (IssueWithQuickFix) issues.get(0);

    assertThat(issue.getQuickFixes()).hasSize(1);

    String codeQFApplied = PythonQuickFixVerifier.applyQuickFix(codeWithIssue, issue);
    assertThat(codeQFApplied).isEqualTo(codeFixed);
  }
}
