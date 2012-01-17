 void palettesimple()
 {
 //example of new colors and definition of a new palette
 
 const Int_t colNum = 20,paletteSize=256;
// const Int_t colNum = 20,paletteSize=128;
// const Int_t colNum = 20,paletteSize=64;
 Int_t startColor=700;
 Int_t palette[paletteSize],n=5;
 Float_t R,G,B;

 //changing default palette
//All grey scale
 R = 1.0; 
 G = 1.0; 
 B = 1.0;
 for(int i = 0; i < paletteSize ; i++) {
   double a;
   a = 1 - ((float)i)/(float)paletteSize;
//   a = 1- ((float) i) / (float)paletteSize;
    TColor *color = new TColor(startColor+i,a*R,a*G,a*B,"");
    palette[i] = startColor+i;
 }

 gStyle->SetPalette(paletteSize,palette);
 

 }

