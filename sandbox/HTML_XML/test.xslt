<?xml version="1.0"?>
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:output method="html"/>
  <xsl:template match="/">
    <HTML>
      <HEAD>
        <TITLE><xsl:value-of select="//summary/heading"/></TITLE>
      </HEAD>
      <BODY>
        <H1><xsl:value-of select="//summary/heading"/></H1>
        <H2><xsl:value-of select="//summary/subhead"/></H2>
        <P><xsl:value-of select="//summary/description"/></P>    
      </BODY>
    </HTML>
  </xsl:template>
</xsl:stylesheet>
