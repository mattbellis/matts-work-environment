#include<cstdio>
#include<cstdlib>
#include<string>
#include<iostream>

using namespace std; // Needed so we don't have to type std:: before string

int main() {

    // In this example we will dynamically (at run time)
    // declare and inititalize an array of C++ string objects.

    // Declare an array of pointers to strings.
    string *mystrings[16];

    char temp[4]; // Dummy variable for some work. Only 4 characters.

    // Loop over them and construct an emptry string for each one
    for(int i=0;i<16;i++) {

        sprintf(temp,"%04d",i); // Make a temporary string

        // We set our string equal to thse characters.
        mystrings[i] = new string(temp);

    }

    // Print them out to see what we got!
    for(int i=0;i<16;i++) {
        cout << *mystrings[i] << endl;
    }

    return 0;

}
