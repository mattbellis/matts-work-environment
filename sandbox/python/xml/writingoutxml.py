#!/usr/bin/env python

import sys
from xml.etree.ElementTree import ElementTree, SubElement

doc = ElementTree(file="memo.xml")
#find the "body" element by tag name
body = doc.getroot()
#Remove all child elements, text (and attributes)
#body.clear()
#Insert new lead text
#body.text = "This is a new meeemmmmmmmmmmmemo.  Send responses to \n"
new_element = SubElement(body, 'entry', {'name': 'thistest'})
new_element.text = "\n\t"
new_element.tail = "\n"
sub0 = SubElement(new_element,"value")
sub0.set('one',"ONE")
sub0.set('two',"TWO")
sub0.set('three',"THREE")
#sub0.append({'two':"TWO"})
sub0.text = "\n"
sub0.tail = "\n"
#write out the modified XML
new_element.tail = "\n"
doc.write("memo.xml")
