/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.plugins.python.pylint;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashSet;
import java.util.Set;
import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;
import org.sonar.api.utils.log.Logger;
import org.sonar.api.utils.log.Loggers;
import org.sonarsource.analyzer.commons.xml.SafetyFactory;

public class PylintRuleParser {

  private static final Logger LOG = Loggers.get(PylintRuleParser.class);
  private Set<String> definedRulesId = new HashSet<>();
  private StringBuilder currentKey = new StringBuilder();

  public PylintRuleParser(String rulesPath) {
    try (InputStream inputStream = getClass().getResourceAsStream(rulesPath)) {
      XMLEventReader reader = SafetyFactory.createXMLInputFactory().createXMLEventReader(inputStream);
      while (reader.hasNext()) {
        onXmlEvent(reader.nextEvent());
      }
    } catch (IOException | XMLStreamException | IllegalArgumentException e) {
      LOG.warn("Unable to parse the Pylint rules definition XML file");
    }

    if (definedRulesId.isEmpty()) {
      LOG.warn("No rule key found for Pylint");
    }
  }

  private void onXmlEvent(XMLEvent event) {
    if (event.isStartElement()) {
      StartElement element = event.asStartElement();
      String elementName = element.getName().getLocalPart();
      if ("key".equals(elementName)) {
        currentKey = new StringBuilder();
      }
    } else if (event.isCharacters()) {
      currentKey.append(event.asCharacters().getData());
    } else if (event.isEndElement() && "key".equals(event.asEndElement().getName().getLocalPart())) {
      definedRulesId.add(currentKey.toString());
    }
  }

  public boolean hasRuleDefinition(String ruleId) {
    return definedRulesId.contains(ruleId);
  }

}
