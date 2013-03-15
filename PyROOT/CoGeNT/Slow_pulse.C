/*
 *  Slow_pulse.c
 *  
 *
 *  Created by Nicole Fields on 6/10/11.
 *  Copyright 2011 University of Chicago. All rights reserved.
 *
 *This is a ROOT (Sigh) script to read in and analyze CoGeNT data.
 *
 * going to try to figure out what the slow pulse correction should be by applying the fast pulse correction to the Energy spectrum from 0.5 to 2.9 keV
 */

{
	
	//ROOT stuff goes here
	
	//gROOT->Reset();
	gROOT->SetStyle("Plain");
	gStyle->SetOptStat(0000000);
	//gROOT->ProcessLine(".x $MGDODIR/Root/LoadMGDOClasses.C");
	// C++ includes...
	#include "iostream"
	#include "string"
	// Root includes...
	#include "TROOT.h"
	#include "TChain.h"
	
	#include "TH1D.h"
	#include "TH2D.h"
	#include "TH3D.h"
	#include "TCanvas.h"
	
	#include "TMath.h"
	#include "TPad.h"
	#include "TStyle.h"
	#include "TFile.h"
	
	#include <stdio.h>
	#include <string.h>
	

	
	//number of bins in the histograms
	int nbins=12;
	//12 bins = 200eV bins 


	FILE *LG=fopen("before_fire_LG.txt","r+");
	FILE *Cosmo=fopen("cosmogenic_data.txt","r+");


	//total exposure time
	int days=458;
	//first day of exposure
	int dayzero=0;
	//number of days in a year
	int daysperyear=365;
	//number of seconds in a day
	int secsperday=86400;
	
	//threshold
	double Emin=0.5;
	
	//max of each of the energy channels
	double LEmax=2.9;
		
	
	
	int i,j;
	int large=100000;
	int  LGlines=0;
	double LGtime[large];
	double LGenergy[large];
	double LGtimeinit;
	
	
	double LGslope=63.7;
	double LGoffset=0.013;
	
	int status;
	
	
	
	TH1D *LGainEnergy = new TH1D("LGainEnergy","LGainEnergy Title (keV)", nbins, Emin, LEmax);
	TH1D *Fastfrachist = new TH1D("Fastfrachist","Fastfrachist Title (keV)", nbins, Emin, LEmax);
	TH1D *LGainEnergyFast = new TH1D("LGainEnergyFast","LGainEnergyFast Title (keV)", nbins, Emin, LEmax);
	TH1D *LGainEnergySlow = new TH1D("LGainEnergySlow","LGainEnergySlow Title (keV)", nbins, Emin, LEmax);
	//TH1D *LGainEnergyFastMin = new TH1D("LGainEnergyFastMin","LGainEnergyFastMin Title (keV)", nbins, Emin, LEmax);
	//TH1D *LGainEnergyFastMax = new TH1D("LGainEnergyFastMax","LGainEnergyFastMax Title (keV)", nbins, Emin, LEmax);
	
	
	
	for(i=0; i<=large; i++){
		//printf("%i\n",i);
		status=fscanf(LG,"%lf %lf",&LGtime[i], &LGenergy[i]);
		
		if (status == EOF) break;
		
		if(i==0){LGtimeinit=LGtime[i];}
		//convert from random time in seconds to time after Dec 4 in days
		LGtime[i]=(LGtime[i]-LGtimeinit)/secsperday;

		//convert from voltage to keV
		//13 eV offset comes from fit to full data set
		LGenergy[i]=LGenergy[i]*LGslope+LGoffset;
		LGainEnergy->Fill(LGenergy[i]);
		
		//printf("%lf %lf\n", LGtime[i], LGenergy[i]);	
		LGlines++;

	}

	
	//make sure the errors are dealt with corectly
	LGainEnergy->Sumw2();
	
	//adjust histogram for slow pulse correction
	
	//values greater than 1 are NOT PHYSICAL
	double Fastfrac[nbins] = {0.33577, 0.52385, 0.70912, 0.84128, 0.78405, 1.0693, 1.0649, 1.0500, 1.0369, 1.0125, 1.0212, 1.0000};
	//double Fastfrac[nbins] = {0.33577, 0.52385, 0.70912, 0.84128, 0.78405, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

	double Fastfracerr[nbins] = {0.046766, 0.077374, 0.093713, 0.087165, 0.10414, 0.24288, 0.23896, 0.25934, 0.25492, 0.22570, 0.25009, 0.25400};
	
	for(i=1; i<(nbins+1);i++){
		

		Fastfrachist->SetBinContent(i, Fastfrac[i-1]);
		Fastfrachist->SetBinError(i, Fastfracerr[i-1]);
		
		
	}
	
	Fastfrachist->Sumw2();

	LGainEnergyFast->Multiply(LGainEnergy, Fastfrachist, 1.0, 1.0);
	LGainEnergySlow->Add(LGainEnergyFast, LGainEnergy, -1.0, 1.0);
	
	TF1 *model=new TF1("model", "expo(0)", Emin, LEmax);
	
	
	//parameters for exponential -> very vague guesses....
	//exponential is of the form exp(p0+p1*x)
	model->SetParameter(0,2.0);
	model->SetParameter(1,-2.0);
	
	LGainEnergySlow->Fit("model");
	
	printf("The total number of slow pulse events in the data passing cuts is %lf.\n", LGainEnergySlow->Integral(1, nbins));
	

	new TCanvas("c2");
	LGainEnergy->GetYaxis()->SetTitle("Counts");
	LGainEnergy->GetXaxis()->SetTitle("keV");
	LGainEnergy->SetLineColor(kRed);
	LGainEnergy->Draw("hist");

	LGainEnergyFast->GetYaxis()->SetTitle("Counts");
	LGainEnergyFast->GetXaxis()->SetTitle("keV");
	LGainEnergyFast->SetLineColor(kBlue);
	LGainEnergyFast->Draw("histsame");
	
	LGainEnergySlow->GetYaxis()->SetTitle("Counts");
	LGainEnergySlow->GetXaxis()->SetTitle("keV");
	LGainEnergySlow->SetLineColor(kGreen);
	LGainEnergySlow->Draw("histsame");
	
	
	
	fclose(LG);

}


