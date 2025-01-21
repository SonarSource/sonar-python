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
package org.sonar.plugins.python.xunit;

import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import javax.xml.stream.XMLStreamException;
import org.codehaus.staxmate.in.ElementFilter;
import org.codehaus.staxmate.in.SMHierarchicCursor;
import org.codehaus.staxmate.in.SMInputCursor;
import org.sonar.api.utils.ParsingUtils;
import org.sonar.plugins.python.parser.StaxParser.XmlStreamHandler;

public class TestSuiteParser implements XmlStreamHandler {

  private List<TestSuite> testSuites = new ArrayList<>();

  @Override
  public void stream(SMHierarchicCursor rootCursor) throws XMLStreamException {
    SMInputCursor testSuiteCursor = rootCursor.constructDescendantCursor(new ElementFilter("testsuite"));
    while (testSuiteCursor.getNext() != null) {
      String testSuiteClassName = getExpectedAttribute(testSuiteCursor, "name");
      TestSuite testSuite = new TestSuite(testSuiteClassName);
      testSuites.add(testSuite);
      SMInputCursor testCaseCursor = testSuiteCursor.childElementCursor("testcase");

      while (testCaseCursor.getNext() != null) {
        testSuite.addTestCase(parseTestCaseTag(testCaseCursor));
      }
    }
  }

  /**
   * Returns successfully parsed reports as a collection of TestSuite objects.
   */
  public Collection<TestSuite> getParsedReports() {
    return testSuites;
  }

  private static TestCase parseTestCaseTag(SMInputCursor testCaseCursor) throws XMLStreamException {
    String name = parseTestCaseName(testCaseCursor);
    Double time = parseTime(testCaseCursor);
    String status = TestCase.STATUS_OK;
    String stack = "";
    String msg = "";

    String file = testCaseCursor.getAttrValue("file");
    String testClassName = testCaseCursor.getAttrValue("classname");

    SMInputCursor childCursor = testCaseCursor.childElementCursor();
    if (childCursor.getNext() != null) {
      String elementName = childCursor.getLocalName();
      if (TestCase.STATUS_SKIPPED.equals(elementName)) {
        status = TestCase.STATUS_SKIPPED;
      } else if (TestCase.STATUS_FAILURE.equals(elementName)) {
        status = TestCase.STATUS_FAILURE;
        msg = getExpectedAttribute(childCursor, "message");
        stack = childCursor.collectDescendantText();
      } else if (TestCase.STATUS_ERROR.equals(elementName)) {
        status = TestCase.STATUS_ERROR;
        msg = getExpectedAttribute(childCursor, "message");
        stack = childCursor.collectDescendantText();
      }
    }
    return new TestCase(name, time.intValue(), status, stack, msg, file, testClassName);
  }

  private static double parseTime(SMInputCursor testCaseCursor) throws XMLStreamException {
    double time;
    try {
      Double tmp = ParsingUtils.parseNumber(getExpectedAttribute(testCaseCursor, "time"), Locale.ENGLISH);
      time = ParsingUtils.scaleValue(tmp * 1000, 3);
    } catch (ParseException e) {
      throw new XMLStreamException(e);
    }
    return time;
  }

  private static String getExpectedAttribute(SMInputCursor testCaseCursor, String attributeName) throws XMLStreamException {
    String attrValue = testCaseCursor.getAttrValue(attributeName);
    if(attrValue == null) {
      throw new IllegalStateException(String.format("Missing attribute '%s' at line %d", attributeName, testCaseCursor.getStreamLocation().getLineNumber()));
    }
    return attrValue;
  }

  private static String parseTestCaseName(SMInputCursor testCaseCursor) throws XMLStreamException {
    String name = getExpectedAttribute(testCaseCursor, "name");
    String classname = testCaseCursor.getAttrValue("CLASSNAME");
    if (classname != null) {
      name = classname + "/" + name;
    }
    return name;
  }

}
