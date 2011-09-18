{
    TRandom rnd;

    // Multiple two 2x2 matrices
    cerr << " ---- Mutlply two 2x2 matrices ---- " << endl;
    TMatrixD a(2,2);
    TMatrixD b(2,2);

    for(i=0;i<2;i++)
    {
        for(j=0;j<2;j++)
        {
            a(i,j) = rnd.Rndm();
            b(i,j) = rnd.Rndm();
        }
    }

    TMatrixD c = a*b;
    a.Print("v");
    b.Print("v");
    c.Print("v");

    // Multiple two 2x2 matrices, but transpose one first
    // First declare a matrix and then you have to fill the elements
    // by referencing the matrix which you want the transpose of.
    cerr << " ---- Transposed matrices ----- " << endl;
    TMatrixD bT(2,2);
    bT.Transpose(b);

    b.Print("v");
    bT.Print("v");

    TMatrixD d = a*bT;
    d.Print("v");


    // Multiple two different matrices
    cerr << " ---- Mutlply a 4x1 by a 1x4 two ways ---- " << endl;
    TMatrixD x(4,1);
    TMatrixD y(1,4);

    for(i=0;i<4;i++)
    {
        for(j=0;j<1;j++)
        {
            x(i,j) = rnd.Rndm();
            y(j,i) = rnd.Rndm();
        }
    }

    TMatrixD z = x*y;
    TMatrixD z2 = y*x;
    x.Print("v");
    y.Print("v");
    z.Print("v");
    z2.Print("v");

    cout << "The first entry for z2 is: " << z2(0,0) << endl;

}
