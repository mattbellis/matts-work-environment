#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<math.h>

using namespace std;

double hbarc = 0.1973269664036767; // in GeV nm 
double c = 3E10; // cm/s 
double rhoDM = 0.3;  // in GeV/cm^3
double mDM = 6.8; // in GeV 
double mn = 0.938; // mass of nucleon in GeV 
double mU = 0.9315; // amu in GeV 
double sigma_n = 1E-40; // in cm^2 
double Na = 6.022E26; // Avogadro's # in mol/kg 
double zeroVec[]={0,0,0};
double AGe = 72.6,ANa = 23,AXe = 131.6; // Atomic masses of targets


void printVec(double* vec)
{
  printf("\n%f %f %f",vec[0],vec[1],vec[2]);
}



double dot(double* v1, double* v2)
{
  return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}


// this function will normalize first vector and put in second vector
void normalize(double* v, double* vnorm)
{
  for(int i=0;i<=2;i++)
    {
      vnorm[i]=v[i]/sqrt(dot(v,v));
    }
}

double mT(double A)
{
  return A*mU;
}


double mu_n(double m)
{
  return (m*mn)/(m + mn);
}


double mu( double A, double m)
{
  return (mT(A)*m)/(mT(A)+m);
}






//minimum speed DM must have to produce recoil of energy Er
// Er is assumed to be in keV, and A is atomic mass of target, m is DM mass
// result is in km/s
double vmin(double Er, double A, double m)
{
  return c*sqrt(Er*mT(A)/2)*1E-8/mu(A,m);
}

//Earth orbit parameters
double t1 = 0.218*365; //  
double e1[] = {0.9931,0.1170,-0.01032}; 
double e2[]={-0.0670, 0.4927, -0.8676}; 
double vorb = 30;
double omega = 2*M_PI/365;


//////////////////
//SHM parameters//
//////////////////
const double vo = 220,vesc = 544; 
// galactic velocity of sun, comes from peculiar velocity and local standard of rest 
double vsVec[]={10,13+vo,7};
double vs=sqrt(dot(vsVec,vsVec));
double vsVecHat[]={vsVec[0]/vs,vsVec[1]/vs,vsVec[2]/vs};
double z=vesc/vo;
double Nesc=erf(z)-2*z*exp(-z*z)/sqrt(M_PI);



// this function will put the total velocity of the earth (observer) in galactic coordinates in the passed vector at the given time
void vObs_t(double* vObs, double t)
{
  for(int i=0; i<=2;i++)
    {
      vObs[i]=vsVec[i]+vorb*(cos(omega*(t-t1))*e1[i]+sin(omega*(t-t1))*e2[i]);
    }
}


// this function will put only the orbital velocity of the earth in galactic coordinates in the passed vector at the given time
void vE_t(double* vE, double t)
{
  for(int i=0; i<=2;i++)
    {
      vE[i]=vorb*(cos(omega*(t-t1))*e1[i]+sin(omega*(t-t1))*e2[i]);
    }
}

// this function is the projection of the observer's velocity onto the stream
// the veoctr that is passes must be the unit vecotr pointing along the stream
// since the earth moves, this function is time dependent
double alpha(double* vstr, double t)
{
  double vObs[]={0,0,0}; 
  vObs_t(vObs,t);
  return dot(vObs,vstr);
}  


// this function will give the magnitude of the stream along the given direction
// that will have a cut-off at the given recoil energy for target atomic number A
// at a time t and for dark matter mass of m 
double vstr(double* vstrHat, double A, double t, double m, double Er)
{
  double vObs[]={0,0,0};
  vObs_t(vObs,t);
  double vObsSqrd=dot(vObs,vObs);
  return alpha(vstrHat, t)+sqrt(alpha(vstrHat, t)*alpha(vstrHat, t)+vmin(Er,A,m)*vmin(Er,A,m)-vObsSqrd);
}


// this function gives the speed of the stream in the frame of the earth
double vstrEarth(double* vstr, double t)
{
  double vObs[]={0,0,0};
  vObs_t(vObs,t);
  return sqrt(dot(vstr,vstr)+dot(vObs,vObs)-2*dot(vstr,vObs));
}


// returns the energy cut-off for a given stream in keV
//t is in days, vstr in km/s, m is DM mass in GeV
double EcutOff(double* vstr, double t,double A, double m)
{
  return 2.0E16*mu(A,m)*mu(A,m)*vstrEarth(vstr,t)*vstrEarth(vstr,t)/(c*c*mT(A));
}

//  return c*sqrt(Er*mT(A)/2)*1E-8/mu(A,m);

// This function finds the time when the stream will have its' max phase
double tc(double* vstr)
{
  double b1,b2,b,t;
  double vSunRel[]={vsVec[0]-vstr[0],vsVec[1]-vstr[1],vsVec[2]-vstr[2]};
  double vSunRelHat[]={0,0,0};
  normalize(vSunRel,vSunRelHat);
  b1=dot(e1,vSunRelHat);
  b2=dot(e2,vSunRelHat);
  b=sqrt(b1*b1+b2*b2);
  t=acos(b1/b)/omega;
  if(b2<0)
    {
      t=2*M_PI/omega-t;
    }
  return t+t1;
}

// Velocity distribution integral for a stream
// vstrE must be stream speed in frame of earth
// vo is the dispersion of the stream i.e. 
// f(v)~exp(-(v-vstrE)^2/v0^2)
double gStream(double vmin, double vstrE, double v0)
{
  return (erf((vmin+vstrE)/v0)-erf((vmin-vstrE)/v0))/(2*vstrE);
}

// Same as above but if the stream has zero dispersion (i.e. delta function)
double gStreamZeroDispersion(double vmin, double vstrE)
{
  if(vmin>vstrE) return 0;
  else return 1/vstrE;
}


//Thse functions are the velocities integrals for the SHM
double glow(double Er, double t, double A, double m)
{
  double vObs[]={0,0,0}; 
  vObs_t(vObs,t);

  double y=sqrt(dot(vObs,vObs))/vo;
  double x=vmin(Er,A,m)/vo;

  //  printf("\nve(t)=%f",vo*y);

  return (erf(x+y)-erf(x-y)-4*y*exp(-z*z)/sqrt(M_PI))/(2*Nesc*vo*y);
}

double ghigh(double Er, double t, double A, double m)
{
  double vObs[]={0,0,0}; 
  vObs_t(vObs,t);

  double y=sqrt(dot(vObs,vObs))/vo;
  double x=vmin(Er,A,m)/vo;

  return (erf(z)-erf(x-y)+2*(x-y-z)*exp(-z*z)/sqrt(M_PI))/(2*Nesc*vo*y);
}
    
double gSHM(double Er, double t, double A, double m)
{
  double vObs[]={0,0,0}; 
  vObs_t(vObs,t);

  double y=sqrt(dot(vObs,vObs))/vo;
  double x=vmin(Er,A,m)/vo;

  if(x<z-y) return glow(Er,t,A,m);
  else if(x<y+z) return ghigh(Er,t,A,m);
  else return 0;
}


// This works ok for overall rates, but will not work well if doing a modulation analysis
double gDebris(double vmin, double vflow, double t)
{
  double vObs[]={0,0,0}; 
  vObs_t(vObs,t);
  double vobs=sqrt(dot(vObs,vObs));
  if(vmin<fabs(vflow-vobs)) return (vflow+vobs-fabs(vflow-vobs))/(2*vflow*vobs);
  else if(vmin<vflow+vobs) return (vflow+vobs-vmin)/(2*vflow*vobs);
  else return 0;
}

//Form Factor Functions
const double a=0.523; const double s=0.9;
double cf(double A)
{
  return 1.23*pow(A,1.0/3.0)-0.6;
}



double q(double Er, double A)
{
  return sqrt(2*mT(A)*Er*1.0E-6);
}

double R1(double A)
{
  return sqrt(cf(A)*cf(A)+7*M_PI*M_PI*a*a/3-5*s*s);
}
double Z1(double Er,double A)
{
  return q(Er,A)*R1(A)/hbarc;
}
double Zs(double Er, double A)
{
  return q(Er,A)*s/hbarc;
}
double j1(double x)
{
  return (sin(x)-x*cos(x))/(x*x);
}
double F( double Er,double A)
{
  return 3*j1(Z1(Er,A))*exp(-Zs(Er,A)*Zs(Er,A)/2)/Z1(Er,A);
}


//All of the particle physics in the spectrum that is not the velocity distribution integral
//This will give the spectra in counts per kg per keV per day 
double spectraParticle(double Er,double A, double m)
{
  return Na*c*c*rhoDM*A*A*sigma_n*F(Er,A)*F(Er,A)*1.0E-11*24*3600/(2*m*mu_n(m)*mu_n(m));
}


// The spectrum for a stream
double dRdErStream(double Er, double t, double A, double* vstr, double v0,double m)
{
  return spectraParticle(Er,A,m)*gStream(vmin(Er,A,m),vstrEarth(vstr,t),v0);
}


// The SHM spectrum
double dRdErSHM(double Er, double t, double A, double m)
{
  return spectraParticle(Er,A,m)*gSHM(Er,t,A,m);
}


// The debris spectrum from Lisanti's paper
double dRdErDebris(double Er,double t, double A, double m, double vflow)
{
  return spectraParticle(Er,A,m)*gDebris(vmin(Er,A,m),vflow,t);
}



int main(int argc,char** argv)
{
  int i;double Er;

  // Find the max phase for the SHM.  This corresponds to a stream with zero velocity

  double tc_SHM=tc(zeroVec);
  printf("\nThe SHM maximum phase occurs at %f days.",tc_SHM);

  printf("\n\nSpectrum for the SHM at t=%f",tc_SHM);
  
  for(i=0;i<=10;i++)
    {
      Er=0.5+i*0.5;
      printf("\n%.2E %.2E",Er,dRdErSHM(Er, tc_SHM, AGe, mDM));
    }


  
 
  //Now find the stream that has the maximum modulation
  //This is a stream that points opposite to the direction of the earth's orbital motion
  //at the time when the SHM has it's max. phase
  double vMaxMod[]={0,0,0};
  double vEWinter[]={0,0,0};

  vE_t(vEWinter,tc_SHM+365./2);
  normalize(vEWinter,vMaxMod);  
  
  printf("\n\n\n\nIn galactic coordinates, the stream with maximum modulation:\nDirection:");
  printVec(vMaxMod);
  double tc_Max=tc(vMaxMod);
  printf("\nMaximum phase at t=%f.\n",tc_Max);

  //Now Choose a maximum modulating stream that has a cut-off at the given energy
  double Er1=3,v01=10; //v01 is the dispersion of this stream
  double vstr1=vstr(vMaxMod,AGe,tc_SHM,mDM,Er1);
  printf("\nStream characteristics for a target with atomic number %.2f and energy cut-off at %f keV:",AGe,Er1);  
  vstr1=vstr(vMaxMod,AGe,153,mDM,Er1);
  double vstr1Vec[]={vstr1*vMaxMod[0],vstr1*vMaxMod[1],vstr1*vMaxMod[2]};


  printf("\nIn galactic coordinates:");
  printf("\nSpeed=%f  Dispersion=%f.",vstr1,v01); 


  printf("\nIn earth's frame,");  

  printf("\nmaximum: Ecutoff=%f stream speed=%f",EcutOff(vstr1Vec, tc_Max,AGe, mDM),vstrEarth(vstr1Vec,tc_Max));
  printf("\nminimum: Ecutoff=%f stream speed=%f",EcutOff(vstr1Vec, tc_Max+365./2.,AGe, mDM),vstrEarth(vstr1Vec,tc_Max+365./2.));

  printf("\nSpectrum for this stream at t=%f",tc_Max);
  
  for(i=0;i<=10;i++)
    {
      Er=2.5+i*0.1;
      printf("\n%.2E %.8E",Er,dRdErStream(Er, tc_Max, AGe, vstr1Vec, 10,mDM));
    }

 

  //The Sagitarius stream may intersect the solar system
  double vSag=300,v0Sag=10;
  double vSagHat[]={0,0.233,-0.970};
  double vSagVec[]={vSag*vSagHat[0],vSag*vSagHat[1],vSag*vSagHat[2]};
  double tc_Sag=tc(vSagVec);

  printf("\n\n\n\nThe Sagitarius Stream has a max. phase at %f.\n",tc_Sag);
  

  printf("\nIn galactic coordinates:");
  printf("\nSpeed=%f  Dispersion=%f.",vSag,v0Sag); 

  printf("\nIn earth's frame,");  

  printf("\nmaximum: Ecutoff=%f stream speed=%f",EcutOff(vSagVec, tc_Sag,AGe, mDM),vstrEarth(vSagVec,tc_Sag));
  printf("\nminimum: Ecutoff=%f stream speed=%f",EcutOff(vSagVec, tc_Sag+365./2.,AGe, mDM),vstrEarth(vSagVec,tc_Sag+365./2.));


  printf("\nSpectrum for this stream at t=%f",tc_Sag);
  for(i=0;i<=10;i++)
    {
      Er=1+i*0.1;
      printf("\n%.2E %.8E",Er,dRdErStream(Er, t1, AGe, vSagVec, v0Sag,mDM));
    }



  double vDeb1=340;
  printf("\n\n\n\nDebris spectrum for %.1f\n",vDeb1);

 
  printf("\nSpectrum for debris at t=%f",tc_SHM);
  for(i=0;i<=10;i++)
    {
      Er=0.5+i*0.5;
      printf("\n%.2E %.8E",Er,dRdErDebris(Er, tc_SHM, AGe, mDM, vDeb1));
    }

  return 0;

}
