function importXML()
{
  if (document.implementation && document.implementation.createDocument)
  {
    xmlDoc = document.implementation.createDocument("", "", null);
    xmlDoc.onload = createTable;
  }
  else if (window.ActiveXObject)
  {
    xmlDoc = new ActiveXObject("Microsoft.XMLDOM");
    xmlDoc.onreadystatechange = function () {
      if (xmlDoc.readyState == 4) createTable()
    };
  }
  else
  {
    alert('Your browser can\'t handle this script');
    return;
  }
  xmlDoc.load("emperors.xml");
}

function createTable()
{
  var x = xmlDoc.getElementsByTagName('emperor');
  var newEl = document.createElement('TABLE');
  newEl.setAttribute('cellPadding',5);
  var tmp = document.createElement('TBODY');
  newEl.appendChild(tmp);
  var row = document.createElement('TR');
  for (j=0;j<x[0].childNodes.length;j++)
  {
    if (x[0].childNodes[j].nodeType != 1) continue;
    var container = document.createElement('TH');
    var theData = document.createTextNode(x[0].childNodes[j].nodeName);
    container.appendChild(theData);
    row.appendChild(container);
  }
  tmp.appendChild(row);
  for (i=0;i<x.length;i++)
  {
    var row = document.createElement('TR');
    for (j=0;j<x[i].childNodes.length;j++)
    {
      if (x[i].childNodes[j].nodeType != 1) continue;
      var container = document.createElement('TD');
      var theData = document.createTextNode(x[i].childNodes[j].firstChild.nodeValue);
      container.appendChild(theData);
      row.appendChild(container);
    }
    tmp.appendChild(row);
  }
  document.getElementById('writeroot').appendChild(newEl);
}

