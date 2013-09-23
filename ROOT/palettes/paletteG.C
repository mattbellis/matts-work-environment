 void paletteG()
 {
 //example of new colors and definition of a new palette
 
// Int_t paletteSize=256;
 const Int_t paletteSize=128;
//Int_t paletteSize=64;
 Int_t startColor=500;
 Int_t palette[paletteSize];
 Float_t R,G,B;

 //changing default palette
//All grey scale
 R = 1.0; 
 G = 1.0; 
 B = 1.0;
// Dark Blue
// R = 0.4; 
// G = 0.0; 
// B = 1.0;
 //
 for(int i = 0; i < paletteSize ; i++) {
   double a;
   a =  1- ((float) i) / (float)paletteSize;
//   cout << a << endl;
    TColor *color = new TColor(startColor+i,a*R,a*G,a*B,"");
    palette[i] = startColor+i;
 }

 gStyle->SetPalette(paletteSize,palette);
 

 }

