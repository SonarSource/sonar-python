/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.plugins.python.parser;

import javax.xml.stream.XMLStreamException;
import org.codehaus.staxmate.in.SMHierarchicCursor;
import org.junit.Test;
import org.sonar.plugins.python.parser.StaxParser.XmlStreamHandler;

public class StaxParserTest {

  @Test
  public void test_XML_with_DTD() throws XMLStreamException {
    StaxParser parser = new StaxParser(getTestHandler());
    parser.parse(getClass().getClassLoader().getResourceAsStream("org/sonar/plugins/python/parser/dtd-test.xml"));
  }

  @Test
  public void test_XML_with_XSD() throws XMLStreamException {
    StaxParser parser = new StaxParser(getTestHandler());
    parser.parse(getClass().getClassLoader().getResourceAsStream("org/sonar/plugins/python/parser/xsd-test.xml"));
  }

  @Test
  public void test_XML_with_XSD_and_ampersand() throws XMLStreamException {
    StaxParser parser = new StaxParser(getTestHandler());
    parser.parse(getClass().getClassLoader().getResourceAsStream("org/sonar/plugins/python/parser/xsd-test-with-entity.xml"));
  }

  private XmlStreamHandler getTestHandler() {
    return new XmlStreamHandler() {
      public void stream(SMHierarchicCursor rootCursor) throws XMLStreamException {
        rootCursor.advance();
        while (rootCursor.getNext() != null) {
        }
      }
    };
  }

}
