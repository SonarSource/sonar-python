all:
	antlr3 PythonLexer.g
	javac -cp /usr/share/java/antlr3-runtime.jar:. TestLexer.java 
	javac -cp /usr/share/java/antlr3-runtime.jar:. TestParser.java 

clean:
	rm *.class
	rm Python.tokens PythonLexer.tokens PythonLexer.java PythonParser.java
