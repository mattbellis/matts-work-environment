{
   // --------- S t y l e ---------------------------
   const Bool_t UsePaperStyle = 0;
   // -----------------------------------------------
   
   TString curDynamicPath( gSystem->GetDynamicPath() );
   gSystem->SetDynamicPath( "../lib:" + curDynamicPath );

   TString curIncludePath(gSystem->GetIncludePath());
   gSystem->SetIncludePath( " -I../include " + curIncludePath );

   // load TMVA shared library created in local release 
   // (not required anymore with the use of rootmaps, but problems with MAC OSX)
   if (TString(gSystem->GetBuildArch()).Contains("macosx") ) gSystem->Load( "libTMVA.1" );

   // welcome the user
	 TMVA::Tools::Instance();
   TMVA::gTools().TMVAWelcomeMessage();
   
#include "tmvaglob.C"
   
   TMVAGlob::SetTMVAStyle(); 
   cout << "TMVAlogon: use \"" << gStyle->GetName() << "\" style [" << gStyle->GetTitle() << "]" << endl;
   cout << endl;
}
