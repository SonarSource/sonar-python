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
package org.sonar.plugins.python;

import java.util.Optional;
import javax.annotation.Nullable;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathFactory;
import org.w3c.dom.Node;

public abstract class XmlReportParser {

  private DocumentBuilder documentBuilder;

  private XPath xPath;

  protected DocumentBuilder getDocumentBuilder() throws ParserConfigurationException {
    if (documentBuilder == null) {
      DocumentBuilderFactory builderFactory = DocumentBuilderFactory.newInstance();
      documentBuilder = builderFactory.newDocumentBuilder();
    }
    return documentBuilder;
  }

  protected XPath getXpath() {
    if (xPath == null) {
      xPath = XPathFactory.newInstance().newXPath();
    }
    return xPath;
  }

  @Nullable
  private static String getAttribute(Node node, String attributeName) {
    Node attribute = node.getAttributes().getNamedItem(attributeName);
    return attribute == null ? null : attribute.getTextContent();
  }

  protected Optional<String> getStringAttribute(Node node, String attributeName) {
    String val = getAttribute(node, attributeName);
    return Optional.ofNullable(val);
  }

  protected boolean getBooleanAttribute(Node node, String attributeName) {
    String val = getAttribute(node, attributeName);
    return "true".equals(val);
  }

  protected Optional<Integer> getIntegerAttribute(Node node, String attributeName) {
    String val = getAttribute(node, attributeName);
    try {
      return Optional.of(Integer.parseInt(val));
    } catch (NumberFormatException e) {
      return Optional.ofNullable(null);
    }
  }

}
