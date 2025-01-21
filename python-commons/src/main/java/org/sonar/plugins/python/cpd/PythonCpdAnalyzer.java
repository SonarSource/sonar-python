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
package org.sonar.plugins.python.cpd;

import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.TokenType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.sonar.api.batch.fs.InputFile;
import org.sonar.api.batch.sensor.SensorContext;
import org.sonar.api.batch.sensor.cpd.NewCpdTokens;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.python.TokenLocation;
import org.sonar.python.api.PythonTokenType;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.caching.CpdSerializer;
import org.sonar.python.tree.TreeUtils;

import static org.sonar.plugins.python.caching.Caching.CPD_TOKENS_CACHE_KEY_PREFIX;
import static org.sonar.plugins.python.caching.Caching.CPD_TOKENS_STRING_TABLE_KEY_PREFIX;

public class PythonCpdAnalyzer {

  private static final Logger LOG = LoggerFactory.getLogger(PythonCpdAnalyzer.class);

  private final SensorContext context;

  public PythonCpdAnalyzer(SensorContext context) {
    this.context = context;
  }

  public void pushCpdTokens(InputFile inputFile, PythonVisitorContext visitorContext) {
    Tree root = visitorContext.rootTree();
    if (root != null) {
      NewCpdTokens cpdTokens = context.newCpdTokens().onFile(inputFile);
      List<Token> tokens = TreeUtils.tokens(root);
      List<Token> tokensToCache = new ArrayList<>();
      for (int i = 0; i < tokens.size(); i++) {
        Token token = tokens.get(i);
        TokenType currentTokenType = token.type();
        TokenType nextTokenType = i + 1 < tokens.size() ? tokens.get(i + 1).type() : GenericTokenType.EOF;
        // INDENT/DEDENT could not be completely ignored during CPD see https://docs.python.org/3/reference/lexical_analysis.html#indentation
        // Just taking into account DEDENT is enough, but because the DEDENT token has an empty value, it's the
        // preceding new line which is added in its place to create a difference
        if (isNewLineWithIndentationChange(currentTokenType, nextTokenType) || !isIgnoredType(currentTokenType)) {
          TokenLocation location = new TokenLocation(token);
          cpdTokens.addToken(location.startLine(), location.startLineOffset(), location.endLine(), location.endLineOffset(), token.value());
          tokensToCache.add(token);
        }
      }
      saveTokensToCache(visitorContext, tokensToCache);
      cpdTokens.save();
    }
  }

  public boolean pushCachedCpdTokens(InputFile inputFile, CacheContext cacheContext) {
    String dataKey = dataCacheKey(inputFile.key());
    String tableKey = stringTableCacheKey(inputFile.key());
    byte[] dataBytes = cacheContext.getReadCache().readBytes(dataKey);
    byte[] tableBytes = cacheContext.getReadCache().readBytes(tableKey);
    if (dataBytes == null || tableBytes == null) {
      return false;
    }

    try {
      List<CpdSerializer.TokenInfo> tokens = CpdSerializer.deserialize(dataBytes, tableBytes);

      NewCpdTokens cpdTokens = context.newCpdTokens().onFile(inputFile);
      tokens.forEach(tokenInfo ->
        cpdTokens.addToken(tokenInfo.startLine, tokenInfo.startLineOffset, tokenInfo.endLine, tokenInfo.endLineOffset, tokenInfo.value));
      cpdTokens.save();
      cacheContext.getWriteCache().copyFromPrevious(dataKey);
      cacheContext.getWriteCache().copyFromPrevious(tableKey);
      return true;
    } catch (IOException e) {
      LOG.warn("Failed to deserialize CPD tokens ({}: {})", e.getClass().getSimpleName(), e.getMessage());
    }

    return false;
  }

  private static void saveTokensToCache(PythonVisitorContext visitorContext, List<Token> tokensToCache) {
    CacheContext cacheContext = visitorContext.cacheContext();
    if (!cacheContext.isCacheEnabled()) {
      return;
    }

    try {
      String fileKey = visitorContext.pythonFile().key();

      CpdSerializer.SerializationResult result = CpdSerializer.serialize(tokensToCache);
      cacheContext.getWriteCache().write(stringTableCacheKey(fileKey), result.stringTable);
      cacheContext.getWriteCache().write(dataCacheKey(fileKey), result.data);
    } catch (Exception e) {
      LOG.warn("Could not write CPD tokens to cache ({}: {})", e.getClass().getSimpleName(), e.getMessage());
    }
  }

  private static boolean isNewLineWithIndentationChange(TokenType currentTokenType, TokenType nextTokenType) {
    return currentTokenType.equals(PythonTokenType.NEWLINE) && nextTokenType.equals(PythonTokenType.DEDENT);
  }

  private static boolean isIgnoredType(TokenType type) {
    return type.equals(PythonTokenType.NEWLINE) ||
      type.equals(PythonTokenType.DEDENT) ||
      type.equals(PythonTokenType.INDENT) ||
      type.equals(GenericTokenType.EOF);
  }

  private static String dataCacheKey(String fileKey) {
    return CPD_TOKENS_CACHE_KEY_PREFIX + fileKey.replace('\\', '/');
  }

  private static String stringTableCacheKey(String fileKey) {
    return CPD_TOKENS_STRING_TABLE_KEY_PREFIX + fileKey.replace('\\', '/');
  }
}
