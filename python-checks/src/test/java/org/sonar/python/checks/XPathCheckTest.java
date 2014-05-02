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

import org.sonar.squidbridge.checks.CheckMessagesVerifier;
import org.junit.Test;
import org.sonar.python.PythonAstScanner;
import org.sonar.squidbridge.api.SourceFile;

import java.io.File;

public class XPathCheckTest {

  private XPathCheck check = new XPathCheck();

  @Test
  public void check() {
    check.xpathQuery = "//STATEMENT";
    check.message = "Avoid statements :)";

    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/xpath.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())
        .next().atLine(1).withMessage("Avoid statements :)")
        .noMore();
  }

  @Test
  public void parseError() {
    check.xpathQuery = "//STATEMENT";

    SourceFile file = PythonAstScanner.scanSingleFile(new File("src/test/resources/checks/parsingError.py"), check);
    CheckMessagesVerifier.verify(file.getCheckMessages())
        .noMore();
  }

}
