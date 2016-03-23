/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
import org.sonar.python.PythonAstScanner;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.checks.CheckMessagesVerifier;

import java.io.File;

public class MissingNewlineAtEndOfFileCheckTest {

  private MissingNewlineAtEndOfFileCheck check = new MissingNewlineAtEndOfFileCheck();

  @Test
  public void missing_new_line() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/missingNewlineAtEndOfFile1.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())
        .next().withMessage(String.format(MissingNewlineAtEndOfFileCheck.MESSAGE, "missingNewlineAtEndOfFile1.py"))
        .noMore();
  }

  @Test
  public void missing_new_line_comment() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/missingNewlineAtEndOfFile2.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())
        .next().withMessage(String.format(MissingNewlineAtEndOfFileCheck.MESSAGE, "missingNewlineAtEndOfFile2.py"))
        .noMore();
  }

  @Test
  public void file_with_new_line() {
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/missingNewlineAtEndOfFile3.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())
        .noMore();
  }

}
