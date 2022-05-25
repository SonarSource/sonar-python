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
package org.sonar.python.quickfix;

import org.junit.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.PythonCheck;

import static org.assertj.core.api.Assertions.assertThat;


public class IssueWithQuickFixTest {

  @Test
  public void test(){
    PythonCheck check = Mockito.mock(PythonCheck.class);
    LocationInFile loc1 = new LocationInFile(null, 1,7,10,10);
    IssueLocation issueLocation = IssueLocation.preciseLocation(loc1, "location");
    IssueWithQuickFix issue = new IssueWithQuickFix(check, issueLocation);
    
    assertThat(issue.getQuickFixes()).isEmpty();

    PythonTextEdit textEdit = new PythonTextEdit(loc1, "This is the replacement text");
    PythonQuickFix quickFix = PythonQuickFix.newQuickFix("New Quickfix")
        .addTextEdit()
          .build();
    
    issue.addQuickFix(quickFix);
    issue.addQuickFix(quickFix);
    
    assertThat(issue.getQuickFixes()).hasSize(2);
    assertThat(issue.getQuickFixes().get(0)).isEqualTo(quickFix);
  }

}
