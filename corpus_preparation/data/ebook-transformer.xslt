<?xml version="1.0" encoding="UTF-8"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output omit-xml-declaration="yes" indent="yes"/>
    <!-- <xsl:strip-space elements="*"/> -->
    
    <xsl:template match="/div">
        
        <bookchapter>
            <xsl:attribute name="title">
                <xsl:value-of select="div[@class='document-titles']//h1[@class='title']" />
            </xsl:attribute>
            
            <!-- ####################### -->
            <xsl:variable name="textsection" select="div[@class='document-body']/div[@class='text']" />
            
            <xsl:for-each select="$textsection/p[count(preceding-sibling::h1) = 0]">
                <paragraph>
                    <xsl:attribute name="number">
                        <xsl:value-of select="span[@class='paranumber']/@title" />
                    </xsl:attribute>
                    
                    <xsl:variable name="paragraph_contents" select="*[not(self::span)]|text()"/>
                    <xsl:for-each select="$paragraph_contents">
                        
                        <xsl:variable name="current_pos" select="position()"/>
                        <xsl:variable name="preceding_pos" select="$paragraph_contents[position() = $current_pos - 1]" />
                        <xsl:variable name="following_pos" select="$paragraph_contents[position() = $current_pos + 1]" />

                        <xsl:choose>
                            <xsl:when test="name() = 'a'">
                                <link>
                                    <xsl:attribute name="reference">
                                        <xsl:value-of select="@id" />
                                    </xsl:attribute>

                                    <xsl:variable name="link_id" select="substring(@href,2)"/>
                                    <xsl:attribute name="id">
                                        <xsl:value-of select="$link_id" />
                                    </xsl:attribute>

                                    <xsl:attribute name="infos">
                                        <xsl:value-of select="string(//p[a[@class='FootnoteSymbol' and @id=$link_id]])" />
                                    </xsl:attribute>

                                    <xsl:if test="$preceding_pos[name()]">
                                        <xsl:value-of select="$preceding_pos" />
                                    </xsl:if>
                                    <xsl:if test="$following_pos[name()]">
                                        <xsl:value-of select="$following_pos" />
                                    </xsl:if>

                                </link>
                            </xsl:when>
                            <xsl:when test="name() = 'em'">
                                <xsl:if test="not($preceding_pos[name()]) and not($following_pos[name()])">
                                    <xsl:value-of select="." />
                                </xsl:if>
                            </xsl:when>
                            <!-- <xsl:when test=".[not(name())]"> -->
                            <xsl:when test="name() = ''">
                                <xsl:value-of select="." />
                            </xsl:when>
                            <xsl:otherwise>
                                <xsl:apply-templates />
                            </xsl:otherwise>
                        </xsl:choose>
                    </xsl:for-each>
                    
                </paragraph>
            </xsl:for-each>

            <!-- ####################### -->
            
            <xsl:for-each select="$textsection/h1">
                
                <paragraph>
                    <xsl:attribute name="title">
                        <xsl:value-of select="strong" />
                    </xsl:attribute>
                    
                    <xsl:variable name="precedingh1count" select="count(preceding-sibling::h1)" />
                    
                    <xsl:for-each select="following-sibling::p[not(img) and count(preceding-sibling::h1) = $precedingh1count + 1]">
                        <xsl:choose>
                            <xsl:when test="@class = 'texte'">
                                <paragraph>
                                    <xsl:attribute name="number">
                                        <xsl:value-of select="span[@class='paranumber']/@title" />
                                    </xsl:attribute>

                                    <!-- ################## -->
                                    <xsl:variable name="paragraph_contents" select="*[not(self::span)]|text()"/>
                                    <xsl:for-each select="$paragraph_contents">

                                        <xsl:variable name="current_pos" select="position()"/>
                                        <xsl:variable name="preceding_pos" select="$paragraph_contents[position() = $current_pos - 1]" />
                                        <xsl:variable name="following_pos" select="$paragraph_contents[position() = $current_pos + 1]" />

                                        <xsl:choose>
                                            <xsl:when test="name() = 'a'">                        
                                                <link>
                                                    <xsl:attribute name="reference">
                                                        <xsl:value-of select="@id" />
                                                    </xsl:attribute>

                                                    <xsl:variable name="link_id" select="substring(@href,2)"/>
                                                    <xsl:attribute name="id">
                                                        <xsl:value-of select="$link_id" />
                                                    </xsl:attribute>

                                                    <xsl:attribute name="infos">
                                                        <xsl:value-of select="string(//p[a[@class='FootnoteSymbol' and @id=$link_id]])" />
                                                    </xsl:attribute>
                                                    
                                                    <xsl:if test="$preceding_pos[name()]">
                                                        <xsl:value-of select="$preceding_pos" />
                                                    </xsl:if>
                                                    <xsl:if test="$following_pos[name()]">
                                                        <xsl:value-of select="$following_pos" />
                                                    </xsl:if>
                                                </link>
                                            </xsl:when>
                                            <xsl:when test="name() = 'em'">
                                                <xsl:if test="not($preceding_pos[name()]) and not($following_pos[name()])">
                                                    <xsl:value-of select="." />
                                                </xsl:if>
                                            </xsl:when>
                                            <!-- <xsl:when test=".[not(name())]"> -->
                                            <xsl:when test="name() = ''">
                                                <xsl:value-of select="." />
                                            </xsl:when>
                                            <xsl:otherwise>
                                                <xsl:copy> <xsl:apply-templates/> </xsl:copy>
                                            </xsl:otherwise>
                                        </xsl:choose>
                                    </xsl:for-each>
                                    <!-- ################## -->

                                </paragraph>
                            </xsl:when>
                            <xsl:when test="@class = 'titreillustration'">
                                <image>
                                    <xsl:attribute name="source">
                                        <xsl:value-of select="following-sibling::p[@class='texte']/img/@src" />
                                    </xsl:attribute>
                                    <xsl:value-of select="." />
                                </image>
                            </xsl:when>
                            <xsl:otherwise>
                            </xsl:otherwise>
                        </xsl:choose>
                    </xsl:for-each>
                </paragraph>
            </xsl:for-each>
        </bookchapter>
    </xsl:template>


    <xsl:template match="span[@class='paranumber']"></xsl:template>
</xsl:stylesheet>