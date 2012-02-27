// @(#)root/eve:$Id: lineset_test.C 26876 2008-12-12 14:45:40Z matevz $
// Author: Matevz Tadel

// Demonstrates usage of class TEveStraightLineSet.

TEveStraightLineSet* hello_eve(Int_t nlines = 40, Int_t nmarkers = 4) 
{
   TEveManager::Create();

   TRandom r(0);
   Float_t s = 100;

   TEveStraightLineSet* ls = new TEveStraightLineSet();

   for(Int_t i = 0; i<nlines; i++)
   {
      ls->AddLine( r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s),
                   r.Uniform(-s,s), r.Uniform(-s,s), r.Uniform(-s,s));
      // add random number of markers
      Int_t nm = Int_t(nmarkers* r.Rndm());
      for(Int_t m = 0; m < nm; m++) {
         ls->AddMarker(i, r.Rndm());
      }

     ls->SetMarkerSize(1.5);
     ls->SetMarkerStyle(4);

     gEve->AddElement(ls);
     gEve->Redraw3D();

     //ls->DestroyElements();
     
     cerr << ls->NumParents() << endl;
     cerr << ls->HasParents() << endl;
     cerr << ls->RemoveParent(*gEve) << endl;
     //gEve->RemoveElement(ls, gEve);
   }



   return ls;
}
