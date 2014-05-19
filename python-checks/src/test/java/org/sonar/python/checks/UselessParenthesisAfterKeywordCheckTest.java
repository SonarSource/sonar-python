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

public class UselessParenthesisAfterKeywordCheckTest {

  @Test
  public void test() throws Exception {
    UselessParenthesisAfterKeywordCheck check = new UselessParenthesisAfterKeywordCheck();
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/uselessParenthesisAfterKeyword.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())
      .next().atLine(2).withMessage("Remove the parentheses after this \"assert\" keyword")
      .next().atLine(5).withMessage("Remove the parentheses after this \"del\" keyword")
      .next().atLine(12).withMessage("Remove the parentheses after this \"if\" keyword")
      .next().atLine(14).withMessage("Remove the parentheses after this \"elif\" keyword")
      .next().atLine(20).withMessage("Remove the parentheses after this \"for\" keyword")
      .next().atLine(23).withMessage("Remove the parentheses after this \"in\" keyword")
      .next().atLine(30).withMessage("Remove the parentheses after this \"raise\" keyword")
      .next().atLine(39).withMessage("Remove the parentheses after this \"return\" keyword")
      .next().atLine(44).withMessage("Remove the parentheses after this \"while\" keyword")
      .next().atLine(48).withMessage("Remove the parentheses after this \"yield\" keyword")
      .next().atLine(57).withMessage("Remove the parentheses after this \"except\" keyword")
      .next().atLine(68).withMessage("Remove the parentheses after this \"not\" keyword")
      .noMore();
  }

}
