/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.python.checks;

import org.junit.Test;
import org.sonar.python.PythonAstScanner;
import org.sonar.squidbridge.api.SourceFile;
import org.sonar.squidbridge.checks.CheckMessagesVerifier;

import java.io.File;

public class TooManyLinesInFileCheckTest {

  @Test
  public void test_negative() {
    SourceFile file = scanFile(3);
    CheckMessagesVerifier.verify(file.getCheckMessages())
      .noMore();
  }

  @Test
  public void test_positive() {
    SourceFile file = scanFile(2);
    CheckMessagesVerifier.verify(file.getCheckMessages())
      .next().withMessage(
        "File \"tooManyLinesInFile.py\" has 3 lines, which is greater than 2 authorized. Split it into smaller files.")
      .noMore();
  }

  private SourceFile scanFile(int maximum) {
    TooManyLinesInFileCheck check = new TooManyLinesInFileCheck();
    check.maximum = maximum;
    return PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/tooManyLinesInFile.py"), check);
  }

}
