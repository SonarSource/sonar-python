files=`find $1 -name "*.py"`
for file in $files 
do
    echo "===== $file ======"
    java -cp /usr/share/java/antlr3-runtime.jar:. TestLexer $file
    java -cp /usr/share/java/antlr3-runtime.jar:. TestParser $file
done
