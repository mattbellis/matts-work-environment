#include<iostream>
#include<cstdio>
#include<cstdlib>

using namespace std;

int main()
{

    system("pwd > pwd.temp");
    FILE *fp = fopen("pwd.temp","r");
    char str[256];
    fscanf(fp,"%s",str);
    fclose(fp);
    system("rm pwd.temp");


    return 0;

}
