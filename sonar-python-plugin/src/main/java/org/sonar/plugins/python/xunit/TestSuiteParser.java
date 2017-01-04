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
package org.sonar.plugins.python.xunit;

import java.io.File;
import java.text.ParseException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import javax.xml.stream.XMLStreamException;
import javax.xml.xpath.XPathConstants;
import org.sonar.api.utils.ParsingUtils;
import org.sonar.plugins.python.PythonReportException;
import org.sonar.plugins.python.XmlReportParser;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class TestSuiteParser extends XmlReportParser {

  private Map<String, TestSuite> testSuites = new HashMap<>();

  public void parse(File xmlFile) {
    try {
      Document xmlDocument = getDocumentBuilder().parse(xmlFile);

      NodeList testsuiteNodes = (NodeList) getXpath().compile("testsuite").evaluate(xmlDocument, XPathConstants.NODESET);

      for (int i = 0; i < testsuiteNodes.getLength(); i++) {
        Element testsuiteNode = (Element) testsuiteNodes.item(i);

        Optional<String> testSuiteClassName = getStringAttribute(testsuiteNode, "name");

        NodeList testcaseNodes = testsuiteNode.getElementsByTagName("testcase");
        for (int j = 0; j < testcaseNodes.getLength(); j++) {
          Element testcaseNode = (Element) testcaseNodes.item(j);
          String testClassName = getClassname(testcaseNode, testSuiteClassName.orElse(null));

          TestSuite report = testSuites.get(testClassName);
          if (report == null) {
            report = new TestSuite(testClassName);
            testSuites.put(testClassName, report);
          }
          report.addTestCase(parseTestCaseElement(testcaseNode));
        }
      }
    } catch (Exception e) {
      throw new PythonReportException(e);
    }
  }

  /**
   * Returns successfully parsed reports as a collection of TestSuite objects.
   */
  public Collection<TestSuite> getParsedReports() {
    return testSuites.values();
  }

  private String getClassname(Node testcaseNode, String defaultClassname) {
    Optional<String> testClassName = getStringAttribute(testcaseNode, "classname");
    return testClassName.isPresent() ? testClassName.get() : defaultClassname;
  }

  private TestCase parseTestCaseElement(Element testcaseNode) throws XMLStreamException {
    // TODO: get a decent grammar for the junit format and check the
    // logic inside this method against it.

    String name = parseTestCaseName(testcaseNode);
    Double time = parseTime(testcaseNode);
    String status = "ok";
    String stack = "";
    String msg = "";

    NodeList children = testcaseNode.getChildNodes();
    for (int i = 0; i < children.getLength(); i++) {
      Node child = children.item(i);
      String elementName = child.getNodeName();

      if ("skipped".equals(elementName)) {
        status = "skipped";
      } else if ("failure".equals(elementName)) {
        status = "failure";
        msg = getStringAttribute(child, "message").orElse(null);
        stack = ((Element) child).getTextContent();
      } else if ("error".equals(elementName)) {
        status = "error";
        msg = getStringAttribute(child, "message").orElse(null);
        stack = ((Element) child).getTextContent();
      }
    }
    return new TestCase(name, time.intValue(), status, stack, msg);
  }

  private double parseTime(Element testcaseNode) {
    double time;
    try {
      String timeValue = getStringAttribute(testcaseNode, "time").orElse(null);
      Double tmp = ParsingUtils.parseNumber(timeValue, Locale.ENGLISH);
      time = ParsingUtils.scaleValue(tmp * 1000, 3);
    } catch (ParseException e) {
      throw new PythonReportException(e);
    }
    return time;
  }

  private String parseTestCaseName(Element testcaseNode) {
    String name = getStringAttribute(testcaseNode, "name").orElse(null);
    String classname = getStringAttribute(testcaseNode, "classname").orElse(null);
    if (classname != null) {
      name = classname + "/" + name;
    }
    return name;
  }

}
