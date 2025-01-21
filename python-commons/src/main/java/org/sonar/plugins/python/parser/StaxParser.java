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
package org.sonar.plugins.python.parser;

import com.ctc.wstx.stax.WstxInputFactory;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import javax.xml.stream.XMLResolver;
import javax.xml.stream.XMLStreamException;
import org.apache.commons.lang.StringUtils;
import org.codehaus.staxmate.SMInputFactory;
import org.codehaus.staxmate.in.SMHierarchicCursor;
import org.sonarsource.analyzer.commons.xml.SafeStaxParserFactory;

/**
 * Class copied from deprecated class StaxParser of sonar-plugin-api.
 */
public class StaxParser {

  private SMInputFactory inf;

  private XmlStreamHandler streamHandler;

  public StaxParser(XmlStreamHandler streamHandler) {
    this.streamHandler = streamHandler;
    WstxInputFactory xmlFactory = (WstxInputFactory) SafeStaxParserFactory.createXMLInputFactory();
    xmlFactory.configureForLowMemUsage();
    xmlFactory.getConfig().setUndeclaredEntityResolver(new UndeclaredEntitiesXMLResolver());
    inf = new SMInputFactory(xmlFactory);
  }

  public void parse(File xmlFile) throws XMLStreamException {
    try (FileInputStream input = new FileInputStream(xmlFile)) {
      parse(input);
    } catch (IOException e) {
      throw new XMLStreamException(e);
    }
  }

  public void parse(InputStream xmlInput) throws XMLStreamException {
    SMHierarchicCursor rootCursor = inf.rootElementCursor(xmlInput);
    try {
      streamHandler.stream(rootCursor);
    } finally {
      rootCursor.getStreamReader().closeCompletely();
    }
  }

  private static class UndeclaredEntitiesXMLResolver implements XMLResolver {

    @Override
    public Object resolveEntity(String arg0, String arg1, String fileName, String undeclaredEntity) throws XMLStreamException {
      String undeclared = undeclaredEntity;
      // avoid problems with XML docs containing undeclared entities.. return the entity under its raw form if not a Unicode expression
      if (StringUtils.startsWithIgnoreCase(undeclaredEntity, "u") && undeclaredEntity.length() == 5) {
        int unicodeCharHexValue = Integer.parseInt(undeclaredEntity.substring(1), 16);
        if (Character.isDefined(unicodeCharHexValue)) {
          undeclared = new String(new char[] {(char) unicodeCharHexValue});
        }
      }
      return undeclared;
    }
  }

  /**
   * Simple interface for handling XML stream to parse.
   */
  @FunctionalInterface
  public interface XmlStreamHandler {
    void stream(SMHierarchicCursor rootCursor) throws XMLStreamException;
  }

}
