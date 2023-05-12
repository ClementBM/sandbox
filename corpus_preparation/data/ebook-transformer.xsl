<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:template match="/div">
<book>
    <title>
        <xsl:value-of select="div[@class='document-titles']/h1[@class='title']"/>
    </title>

    <xsl:for-each select="div[@class='document-body']/div[@class='text']/p">

        <xsl:value-of select="/@class"/>

    </xsl:for-each>
</book>
</xsl:template>
</xsl:stylesheet>