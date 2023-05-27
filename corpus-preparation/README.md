

- [Cleaning and formatting data](#cleaning-and-formatting-data)
  - [Text structure](#text-structure)
  - [XSL syntax](#xsl-syntax)
    - [strip-space](#strip-space)
  - [XSLT VScode extensions](#xslt-vscode-extensions)
    - [Run XSL transformation](#run-xsl-transformation)
    - [Debug XSL transformation](#debug-xsl-transformation)
  - [XSL with python, lxml](#xsl-with-python-lxml)
  - [XSLT resources](#xslt-resources)
- [Preprocessing data](#preprocessing-data)
  - [NLTK Corpus](#nltk-corpus)
- [Combining data](#combining-data)


# Cleaning and formatting data
This includes tasks such as handling missing values or outliers, ensuring data is in the correct format, and removing unneeded columns.


## Text structure

```xml
<book>


</book>
```

## XSL syntax

### strip-space
Cette instruction permet de ne pas tenir compte des éléments textes du document XML source ne contenant que des "caractères blancs" (espaces, tabulations, retour-charriots).

```xml
<xsl:strip-space elements="*"/>
```

http://fsajous.free.fr/xml/xslt/instr_strip-space.html

## XSLT VScode extensions
XSLT and XPath

### Run XSL transformation
https://deltaxml.github.io/vscode-xslt-xpath/run-xslt.html

### Debug XSL transformation
https://deltaxml.github.io/vscode-xslt-xpath/code-diagnostics.html

## XSL with python, lxml
Due to a bug in libxslt the usage of <xsl:strip-space elements="*"/> in an XSLT stylesheet can lead to crashes or memory failures. It is therefore advised not to use xsl:strip-space in stylesheets used with lxml.

https://lxml.de/xpathxslt.html#xslt

## XSLT resources
* https://edutechwiki.unige.ch/fr/Tutoriel_XSLT_d%C3%A9butant#Variables_et_templates_avec_param%C3%A8tres
* http://i-like-robots.github.io/xslt-fiddle/
* https://www.w3schools.com/xml/xsl_choose.asp

# Preprocessing data
This includes tasks like numerical transformations, aggregating data, encoding text or image data, and creating new features.

## NLTK Corpus
https://www.nltk.org/howto/corpus.html

# Combining data
This includes tasks like joining tables or merging datasets.