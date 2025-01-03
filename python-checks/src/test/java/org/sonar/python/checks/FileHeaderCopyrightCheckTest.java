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
package org.sonar.python.checks;

import java.io.File;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import org.assertj.core.api.Assertions;
import org.junit.Assert;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.python.SubscriptionVisitor;
import org.sonar.python.TestPythonVisitorRunner;
import org.sonar.python.checks.utils.PythonCheckVerifier;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FileHeaderCopyrightCheckTest {

  @Test
  void test_copyright() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "# Copyright FOO\n";
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/copyright.py", fileHeaderCopyrightCheck);
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/copyrightAndComments.py", fileHeaderCopyrightCheck);
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/commentAndDocstring.py", fileHeaderCopyrightCheck);
  }

  @Test
  void test_noncompliant() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "# Copyright FOO";
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/copyrightNonCompliant.py", fileHeaderCopyrightCheck);
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/noHeaderNonCompliant.py", fileHeaderCopyrightCheck);
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/emptyFileButCopyright.py", fileHeaderCopyrightCheck);
  }

  @Test
  void test_NoCopyright() {
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/headerNoCopyright.py", new FileHeaderCopyrightCheck());
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/emptyFileNoCopyright.py", new FileHeaderCopyrightCheck());
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/emptyFileWithLineBreakNoCopyright.py", new FileHeaderCopyrightCheck());
  }


  @Test
  void test_copyright_docstring() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = """
      ""\"
       SonarQube, open source software quality management tool.
       Copyright (C) 2008-2018 SonarSource
       mailto:contact AT sonarsource DOT com
      
       SonarQube is free software; you can redistribute it and/or
       modify it under the terms of the GNU Lesser General Public
       License as published by the Free Software Foundation; either
       version 3 of the License, or (at your option) any later version.
      
       SonarQube is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
       Lesser General Public License for more details.
      
       You should have received a copy of the GNU Lesser General Public License
       along with this program; if not, write to the Free Software Foundation,
       Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
      ""\"""";
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/docstring.py", fileHeaderCopyrightCheck);
  }

  @Test
  void test_copyright_docstring_noncompliant() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = """
      ""\"
       SonarQube, open source software quality management tool.
       Copyright (C) 2008-2018 SonarSource
       mailto:contact AT sonarsource DOT com
      
       SonarQube is free software; you can redistribute it and/or
       modify it under the terms of the GNU Lesser General Public
       License as published by the Free Software Foundation; either
       version 3 of the License, or (at your option) any later version.
      
       SonarQube is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
       Lesser General Public License for more details.
      
       You should have received a copy of the GNU Lesser General Public License
       along with this program; if not, write to the Free Software Foundation,
       Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
      ""\"""";
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/docstringNonCompliant.py", fileHeaderCopyrightCheck);

    List<PythonCheck.PreciseIssue> issues = PythonCheckVerifier.issues("src/test/resources/checks/fileHeaderCopyright/emptyFileNoCopyright.py", fileHeaderCopyrightCheck);
    assertEquals(1, issues.size());
  }

  @Test
  void test_searchPattern() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "^#\\sCopyright[ ]20[0-9]{2}\\n#\\sAll rights reserved\\.\\n";
    fileHeaderCopyrightCheck.isRegularExpression = true;
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/copyrightNonCompliant.py", fileHeaderCopyrightCheck);
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/searchPatternNonCompliant.py", fileHeaderCopyrightCheck);
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/searchPattern.py", fileHeaderCopyrightCheck);

    List<PythonCheck.PreciseIssue> issues = PythonCheckVerifier.issues("src/test/resources/checks/fileHeaderCopyright/emptyFileNoCopyright.py",
      fileHeaderCopyrightCheck);
    assertEquals(1, issues.size());
  }

  @Test
  void test_misplaced_copyright_searchPattern() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.isRegularExpression = true;
    fileHeaderCopyrightCheck.headerFormat = "Copyright[ ]20[0-9]{2}";
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/searchPatternMisplacedCopyright.py", fileHeaderCopyrightCheck);
  }

  @Test
  void test_searchPattern_exception() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "**";
    fileHeaderCopyrightCheck.isRegularExpression = true;
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(new File("src/test/resources/checks/fileHeaderCopyright/searchPatternThrowsError.py"));
    Collection<PythonSubscriptionCheck> check = Collections.singletonList(fileHeaderCopyrightCheck);

    Assertions.assertThatThrownBy(() -> SubscriptionVisitor.analyze(check, context)).isInstanceOf(IllegalArgumentException.class);

    IllegalArgumentException e = Assert.assertThrows(IllegalArgumentException.class, () -> SubscriptionVisitor.analyze(check, context));
    Assertions.assertThat(e.getMessage()).isEqualTo("[FileHeaderCopyrightCheck] Unable to compile the regular expression: "+fileHeaderCopyrightCheck.headerFormat);
  }

  @Test
  void shebangTest() {
    var fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "# Copyright FOO\n";
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/shebangCopyright.py", fileHeaderCopyrightCheck);
  }

  @Test
  void shebangPatternTest() {
    var fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.isRegularExpression = true;
    fileHeaderCopyrightCheck.headerFormat = "# Copyright[ ]20[0-9]{2}";
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/searchPatternShebangCopyright.py",
      fileHeaderCopyrightCheck);
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/searchPatternShebangCopyrightNonCompliant.py",
      fileHeaderCopyrightCheck);
  }
}
