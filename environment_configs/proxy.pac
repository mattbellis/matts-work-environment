function FindProxyForURL(url, host)
{
/* variable strings to return */
proxy_yes = "SOCKS 127.0.0.1:8080";
proxy_no = "DIRECT";

if (shExpMatch(url, "https://webscs02.slac.stanford.edu*"))
    return proxy_yes;
if (shExpMatch(url, "*jap.aip.org*"))
    return proxy_yes;
if (shExpMatch(url, "*elsevier*"))
    return proxy_yes;
if (shExpMatch(url, "*slaconly*"))
    return proxy_yes;
if (shExpMatch(url, "*springerlink*"))
    return proxy_yes;
if (shExpMatch(url, "*geoscienceworld*"))
    return proxy_yes;
if (shExpMatch(url, "*prd*"))
    return proxy_yes;
if (shExpMatch(url, "*prl*"))
    return proxy_yes;
if (shExpMatch(url, "*ajp*"))
    return proxy_yes;
if (shExpMatch(url, "*nature*"))
    return proxy_yes;
if (shExpMatch(url, "*sciencedirect*"))
    return proxy_yes;
if (shExpMatch(url, "*sciencemag*"))
    return proxy_yes;
if (shExpMatch(url, "*oraweb*"))
    return proxy_yes;
if (shExpMatch(url, "*tisint*"))
    return proxy_yes;
//if (shExpMatch(url, "*bbr*"))
    //return proxy_yes;
if (shExpMatch(url, "*babar-internal*"))
    return proxy_yes;
if (shExpMatch(url, "*internal.slac*"))
    return proxy_yes;
if (shExpMatch(url, "*tislnx*"))
    return proxy_yes;
if (shExpMatch(url, "*oraweb*"))
    return proxy_yes;
if (shExpMatch(url, "*www-group*"))
    return proxy_yes;

return proxy_no;
}


