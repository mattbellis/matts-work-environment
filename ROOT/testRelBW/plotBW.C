{

  TF1 bw("bw","1/(pow(x-1.5,2) + (pow(0.0151,2)/4.))",1.4,1.6);

  TF1 rbw("rbw","9/(pow(pow(x,2)-pow(1.5,2),2) + (pow(1.5*0.0151,2)))",1.4,1.6);
  rbw.SetLineColor(2);
  
  bw.Draw();
  rbw.Draw("same");


}
