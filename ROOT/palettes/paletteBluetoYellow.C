 void paletteBluetoYellow()
 {
 //example of new colors and definition of a new palette

 cerr << "Using palletteBluetoYellow" << endl;
 
 const Int_t colNum = 100,paletteSize=100;
// const Int_t colNum = 20,paletteSize=128;
// const Int_t colNum = 20,paletteSize=64;
 Int_t startColor=700;
 Int_t palette[paletteSize],n=5;
 Float_t R,G,B;

 //changing default palette
//All grey scale
 R = 1.00; 
 G = 1.00; 
 B = 1.00;
// R = 40.0; 
// G = 1.0; 
// B = 30.0;
// R = 1.0; 
// G = 1.0; 
// B = 30.0;
 for(int i = 0; i < paletteSize ; i++) {
   double a,b,c;
//   a = 1- ((float) i) / (float)paletteSize;
   a = 1 - pow( ((float) i) / (float)paletteSize, 0.75);
   b = 1 - pow( ((float) i) / (float)paletteSize, 0.75);
//   c = 1 - 0.5( ((float) i) / (float)paletteSize);
   c = 0.5;
//cerr << c << endl;
    TColor *color = new TColor(startColor+i,a*R,b*G,c*B,"");
    palette[i] = startColor+i;
 }


 gStyle->SetPalette(paletteSize,palette);
 

 }

