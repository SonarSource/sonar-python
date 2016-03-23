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

public class UselessParenthesisCheckTest {

  @Test
  public void test() {
    UselessParenthesisCheck check = new UselessParenthesisCheck();
    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/uselessParenthesis.py"), check);
    String message = "Remove those useless parentheses";
    CheckMessagesVerifier.verify(file.getCheckMessages())
      .next().atLine(1).withMessage(message)
      .next().atLine(6).withMessage(message)
      .next().atLine(10).withMessage(message)
      .next().atLine(12).withMessage(message)
      .next().atLine(19).withMessage(message)
      .next().atLine(21).withMessage(message)
      .next().atLine(27).withMessage(message)
      .next().atLine(30).withMessage(message)
      .next().atLine(37).withMessage(message)
      .next().atLine(39).withMessage(message)
      .next().atLine(44).withMessage(message)
      .next().atLine(55).withMessage(message)
      .next().atLine(66).withMessage(message)
      .next().atLine(69).withMessage(message)
      .next().atLine(73).withMessage(message)
      .noMore();
  }

}
