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

import org.junit.Test;
import org.sonar.python.checks.utils.PythonCheckVerifier;

public class FileHeaderCopyrightCheckTest {

  @Test
  public void test() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "Copyright FOO\n";
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/fileCopyright_compliant.py", fileHeaderCopyrightCheck);
  }

  @Test
  public void test_noncompliant() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "Copyright FOO";
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/fileCopyright_Noncompliant.py", fileHeaderCopyrightCheck);
  }

  @Test
  public void test_noncompliant_noheader() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "Copyright FOO";
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/fileCopyright_NoHeader_NonCompliant.py", fileHeaderCopyrightCheck);
  }

  @Test
  public void test_copyright_comment() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "\n" +
      "SonarQube, open source software quality management tool.\n" +
      "Copyright (C) 2008-2018 SonarSource\n" +
      "mailto:contact AT sonarsource DOT com\n" +
      "\n" +
      "SonarQube is free software; you can redistribute it and/or\n" +
      "modify it under the terms of the GNU Lesser General Public\n" +
      "License as published by the Free Software Foundation; either\n" +
      "version 3 of the License, or (at your option) any later version.\n" +
      "\n" +
      "SonarQube is distributed in the hope that it will be useful,\n" +
      "but WITHOUT ANY WARRANTY; without even the implied warranty of\n" +
      "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU\n" +
      "Lesser General Public License for more details.\n" +
      "\n" +
      "You should have received a copy of the GNU Lesser General Public License\n" +
      "along with this program; if not, write to the Free Software Foundation,\n" +
      "Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.\n" +
      "\n";
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/fileCopyrightComment.py", fileHeaderCopyrightCheck);
  }

  @Test
  public void test_copyrightWithComment() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "Copyright FOO\n";
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/fileCopyrightAndComments.py", fileHeaderCopyrightCheck);
  }

  @Test
  public void test_copyrightWithCommentSpaced() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "Copyright FOO\n";
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/fileCopyrightAndCommentsSpaced.py", fileHeaderCopyrightCheck);
  }

  @Test
  public void test_copyright_docstring() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "\n" +
      "SonarQube, open source software quality management tool.\n" +
      "Copyright (C) 2008-2018 SonarSource\n" +
      "mailto:contact AT sonarsource DOT com\n" +
      "\n" +
      "SonarQube is free software; you can redistribute it and/or\n" +
      "modify it under the terms of the GNU Lesser General Public\n" +
      "License as published by the Free Software Foundation; either\n" +
      "version 3 of the License, or (at your option) any later version.\n" +
      "\n" +
      "SonarQube is distributed in the hope that it will be useful,\n" +
      "but WITHOUT ANY WARRANTY; without even the implied warranty of\n" +
      "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU\n" +
      "Lesser General Public License for more details.\n" +
      "\n" +
      "You should have received a copy of the GNU Lesser General Public License\n" +
      "along with this program; if not, write to the Free Software Foundation,\n" +
      "Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.\n";

    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/fileCopyrightDocstring.py", fileHeaderCopyrightCheck);
  }

  @Test
  public void test_copyright_docstring_noncompliant() {
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "\n" +
      "SonarQube, open source software quality management tool.\n" +
      "Copyright (C) 2008-2018 SonarSource\n" +
      "mailto:contact AT sonarsource DOT com\n";
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/fileCopyrightDocstring_NonCompliant.py", fileHeaderCopyrightCheck);
  }

  @Test
  public void test_emptyFileNoCopyright(){
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/fileCopyright_emptyFileNoCopyright.py", new FileHeaderCopyrightCheck());
  }

  @Test
  public void test_emptyFileButCopyright(){
    FileHeaderCopyrightCheck fileHeaderCopyrightCheck = new FileHeaderCopyrightCheck();
    fileHeaderCopyrightCheck.headerFormat = "Copyright 2022\n";
    PythonCheckVerifier.verify("src/test/resources/checks/fileHeaderCopyright/fileCopyright_emptyFileButCopyright.py", fileHeaderCopyrightCheck);
  }

  @Test
  public void test_NoCopyright(){
    PythonCheckVerifier.verifyNoIssue("src/test/resources/checks/fileHeaderCopyright/fileCopyright_None.py", new FileHeaderCopyrightCheck());
  }

}
