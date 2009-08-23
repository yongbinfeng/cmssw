/*
 * \file EBLaserClient.cc
 *
 * $Date: 2009/08/21 11:52:28 $
 * $Revision: 1.248 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <memory>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <math.h>

#include "DQMServices/Core/interface/DQMStore.h"

#include "OnlineDB/EcalCondDB/interface/MonLaserBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonLaserRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNBlueDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNGreenDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNIRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNRedDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserBlueCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserGreenCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserIRedCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/MonTimingLaserRedCrystalDat.h"
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "CondTools/Ecal/interface/EcalErrorDictionary.h"

#include "DQM/EcalCommon/interface/EcalErrorMask.h"
#include "DQM/EcalCommon/interface/UtilsClient.h"
#include "DQM/EcalCommon/interface/LogicID.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include <DQM/EcalBarrelMonitorClient/interface/EBLaserClient.h>

using namespace cms;
using namespace edm;
using namespace std;

EBLaserClient::EBLaserClient(const ParameterSet& ps) {

  // cloneME switch
  cloneME_ = ps.getUntrackedParameter<bool>("cloneME", true);

  // verbose switch
  verbose_ = ps.getUntrackedParameter<bool>("verbose", true);

  // debug switch
  debug_ = ps.getUntrackedParameter<bool>("debug", false);

  // prefixME path
  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  // enableCleanup_ switch
  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // vector of selected Super Modules (Defaults to all 36).
  superModules_.reserve(36);
  for ( unsigned int i = 1; i <= 36; i++ ) superModules_.push_back(i);
  superModules_ = ps.getUntrackedParameter<vector<int> >("superModules", superModules_);

  laserWavelengths_.reserve(4);
  for ( unsigned int i = 1; i <= 4; i++ ) laserWavelengths_.push_back(i);
  laserWavelengths_ = ps.getUntrackedParameter<vector<int> >("laserWavelengths", laserWavelengths_);

  if ( verbose_ ) {
    cout << " Laser wavelengths:" << endl;
    for ( unsigned int i = 0; i < laserWavelengths_.size(); i++ ) {
      cout << " " << laserWavelengths_[i];
    }
    cout << endl;
  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;
    h05_[ism-1] = 0;
    h06_[ism-1] = 0;
    h07_[ism-1] = 0;
    h08_[ism-1] = 0;

    h09_[ism-1] = 0;
    h10_[ism-1] = 0;
    h11_[ism-1] = 0;
    h12_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;
    hs03_[ism-1] = 0;
    hs04_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;
    i05_[ism-1] = 0;
    i06_[ism-1] = 0;
    i07_[ism-1] = 0;
    i08_[ism-1] = 0;

    i09_[ism-1] = 0;
    i10_[ism-1] = 0;
    i11_[ism-1] = 0;
    i12_[ism-1] = 0;
    i13_[ism-1] = 0;
    i14_[ism-1] = 0;
    i15_[ism-1] = 0;
    i16_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    meg01_[ism-1] = 0;
    meg02_[ism-1] = 0;
    meg03_[ism-1] = 0;
    meg04_[ism-1] = 0;

    meg05_[ism-1] = 0;
    meg06_[ism-1] = 0;
    meg07_[ism-1] = 0;
    meg08_[ism-1] = 0;
    meg09_[ism-1] = 0;
    meg10_[ism-1] = 0;
    meg11_[ism-1] = 0;
    meg12_[ism-1] = 0;

    mea01_[ism-1] = 0;
    mea02_[ism-1] = 0;
    mea03_[ism-1] = 0;
    mea04_[ism-1] = 0;

    met01_[ism-1] = 0;
    met02_[ism-1] = 0;
    met03_[ism-1] = 0;
    met04_[ism-1] = 0;

    metav01_[ism-1] = 0;
    metav02_[ism-1] = 0;
    metav03_[ism-1] = 0;
    metav04_[ism-1] = 0;

    metrms01_[ism-1] = 0;
    metrms02_[ism-1] = 0;
    metrms03_[ism-1] = 0;
    metrms04_[ism-1] = 0;

    meaopn01_[ism-1] = 0;
    meaopn02_[ism-1] = 0;
    meaopn03_[ism-1] = 0;
    meaopn04_[ism-1] = 0;

    mepnprms01_[ism-1] = 0;
    mepnprms02_[ism-1] = 0;
    mepnprms03_[ism-1] = 0;
    mepnprms04_[ism-1] = 0;
    mepnprms05_[ism-1] = 0;
    mepnprms06_[ism-1] = 0;
    mepnprms07_[ism-1] = 0;
    mepnprms08_[ism-1] = 0;

    me_hs01_[ism-1] = 0;
    me_hs02_[ism-1] = 0;
    me_hs03_[ism-1] = 0;
    me_hs04_[ism-1] = 0;

  }

  percentVariation_ = 0.4;

  amplitudeThreshold_ = 100.;

  rmsThresholdRelative_ = 0.3;

  amplitudeThresholdPnG01_ = 50.;
  amplitudeThresholdPnG16_ = 50.;

  pedPnExpectedMean_[0] = 750.0;
  pedPnExpectedMean_[1] = 750.0;

  pedPnDiscrepancyMean_[0] = 100.0;
  pedPnDiscrepancyMean_[1] = 100.0;

  pedPnRMSThreshold_[0] = 1.0; // value at h4; expected nominal: 0.5
  pedPnRMSThreshold_[1] = 3.0; // value at h4; expected nominal: 1.6

}

EBLaserClient::~EBLaserClient() {

}

void EBLaserClient::beginJob(DQMStore* dqmStore) {

  dqmStore_ = dqmStore;

  if ( debug_ ) cout << "EBLaserClient: beginJob" << endl;

  ievt_ = 0;
  jevt_ = 0;

}

void EBLaserClient::beginRun(void) {

  if ( debug_ ) cout << "EBLaserClient: beginRun" << endl;

  jevt_ = 0;

  this->setup();

}

void EBLaserClient::endJob(void) {

  if ( debug_ ) cout << "EBLaserClient: endJob, ievt = " << ievt_ << endl;

  this->cleanup();

}

void EBLaserClient::endRun(void) {

  if ( debug_ ) cout << "EBLaserClient: endRun, jevt = " << jevt_ << endl;

  this->cleanup();

}

void EBLaserClient::setup(void) {

  char histo[200];

  dqmStore_->setCurrentFolder( prefixME_ + "/EBLaserClient" );

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L1 %s", Numbers::sEB(ism).c_str());
      meg01_[ism-1] = dqmStore_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meg01_[ism-1]->setAxisTitle("ieta", 1);
      meg01_[ism-1]->setAxisTitle("iphi", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L2 %s", Numbers::sEB(ism).c_str());
      meg02_[ism-1] = dqmStore_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meg02_[ism-1]->setAxisTitle("ieta", 1);
      meg02_[ism-1]->setAxisTitle("iphi", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L3 %s", Numbers::sEB(ism).c_str());
      meg03_[ism-1] = dqmStore_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meg03_[ism-1]->setAxisTitle("ieta", 1);
      meg03_[ism-1]->setAxisTitle("iphi", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( meg04_[ism-1] ) dqmStore_->removeElement( meg04_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L4 %s", Numbers::sEB(ism).c_str());
      meg04_[ism-1] = dqmStore_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meg04_[ism-1]->setAxisTitle("ieta", 1);
      meg04_[ism-1]->setAxisTitle("iphi", 2);
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L1 PNs G01 %s", Numbers::sEB(ism).c_str());
      meg05_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg05_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg05_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( meg06_[ism-1] ) dqmStore_->removeElement( meg06_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L2 PNs G01 %s", Numbers::sEB(ism).c_str());
      meg06_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg06_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg06_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( meg07_[ism-1] ) dqmStore_->removeElement( meg07_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L3 PNs G01 %s", Numbers::sEB(ism).c_str());
      meg07_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg07_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg07_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( meg08_[ism-1] ) dqmStore_->removeElement( meg08_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L4 PNs G01 %s", Numbers::sEB(ism).c_str());
      meg08_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg08_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg08_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( meg09_[ism-1] ) dqmStore_->removeElement( meg09_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L1 PNs G16 %s", Numbers::sEB(ism).c_str());
      meg09_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg09_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg09_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( meg10_[ism-1] ) dqmStore_->removeElement( meg10_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L2 PNs G16 %s", Numbers::sEB(ism).c_str());
      meg10_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg10_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg10_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( meg11_[ism-1] ) dqmStore_->removeElement( meg11_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L3 PNs G16 %s", Numbers::sEB(ism).c_str());
      meg11_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg11_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg11_[ism-1]->setAxisTitle("channel", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( meg12_[ism-1] ) dqmStore_->removeElement( meg12_[ism-1]->getName() );
      sprintf(histo, "EBLT laser quality L4 PNs G16 %s", Numbers::sEB(ism).c_str());
      meg12_[ism-1] = dqmStore_->book2D(histo, histo, 10, 0., 10., 1, 0., 5.);
      meg12_[ism-1]->setAxisTitle("pseudo-strip", 1);
      meg12_[ism-1]->setAxisTitle("channel", 2);
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );;
      sprintf(histo, "EBLT amplitude L1 %s", Numbers::sEB(ism).c_str());
      mea01_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      mea01_[ism-1]->setAxisTitle("channel", 1);
      mea01_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
      sprintf(histo, "EBLT amplitude L2 %s", Numbers::sEB(ism).c_str());
      mea02_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      mea02_[ism-1]->setAxisTitle("channel", 1);
      mea02_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( mea03_[ism-1] ) dqmStore_->removeElement( mea03_[ism-1]->getName() );
      sprintf(histo, "EBLT amplitude L3 %s", Numbers::sEB(ism).c_str());
      mea03_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      mea03_[ism-1]->setAxisTitle("channel", 1);
      mea03_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( mea04_[ism-1] ) dqmStore_->removeElement( mea04_[ism-1]->getName() );
      sprintf(histo, "EBLT amplitude L4 %s", Numbers::sEB(ism).c_str());
      mea04_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      mea04_[ism-1]->setAxisTitle("channel", 1);
      mea04_[ism-1]->setAxisTitle("amplitude", 2);
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing L1 %s", Numbers::sEB(ism).c_str());
      met01_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      met01_[ism-1]->setAxisTitle("channel", 1);
      met01_[ism-1]->setAxisTitle("jitter", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing L2 %s", Numbers::sEB(ism).c_str());
      met02_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      met02_[ism-1]->setAxisTitle("channel", 1);
      met02_[ism-1]->setAxisTitle("jitter", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( met03_[ism-1] ) dqmStore_->removeElement( met03_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing L3 %s", Numbers::sEB(ism).c_str());
      met03_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      met03_[ism-1]->setAxisTitle("channel", 1);
      met03_[ism-1]->setAxisTitle("jitter", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( met04_[ism-1] ) dqmStore_->removeElement( met04_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing L4 %s", Numbers::sEB(ism).c_str());
      met04_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      met04_[ism-1]->setAxisTitle("channel", 1);
      met04_[ism-1]->setAxisTitle("jitter", 2);
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( metav01_[ism-1] ) dqmStore_->removeElement( metav01_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing mean L1 %s", Numbers::sEB(ism).c_str());
      metav01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      metav01_[ism-1]->setAxisTitle("mean", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( metav02_[ism-1] ) dqmStore_->removeElement( metav02_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing mean L2 %s", Numbers::sEB(ism).c_str());
      metav02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      metav02_[ism-1]->setAxisTitle("mean", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( metav03_[ism-1] ) dqmStore_->removeElement( metav03_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing mean L3 %s", Numbers::sEB(ism).c_str());
      metav03_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      metav03_[ism-1]->setAxisTitle("mean", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( metav04_[ism-1] ) dqmStore_->removeElement( metav04_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing mean L4 %s", Numbers::sEB(ism).c_str());
      metav04_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      metav04_[ism-1]->setAxisTitle("mean", 1);
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( metrms01_[ism-1] ) dqmStore_->removeElement( metrms01_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing rms L1 %s", Numbers::sEB(ism).c_str());
      metrms01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
      metrms01_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( metrms02_[ism-1] ) dqmStore_->removeElement( metrms02_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing rms L2 %s", Numbers::sEB(ism).c_str());
      metrms02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
      metrms02_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( metrms03_[ism-1] ) dqmStore_->removeElement( metrms03_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing rms L3 %s", Numbers::sEB(ism).c_str());
      metrms03_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
      metrms03_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( metrms04_[ism-1] ) dqmStore_->removeElement( metrms04_[ism-1]->getName() );
      sprintf(histo, "EBLT laser timing rms L4 %s", Numbers::sEB(ism).c_str());
      metrms04_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 0.5);
      metrms04_[ism-1]->setAxisTitle("rms", 1);
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( meaopn01_[ism-1] ) dqmStore_->removeElement( meaopn01_[ism-1]->getName() );
      sprintf(histo, "EBLT amplitude over PN L1 %s", Numbers::sEB(ism).c_str());
      meaopn01_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      meaopn01_[ism-1]->setAxisTitle("channel", 1);
      meaopn01_[ism-1]->setAxisTitle("amplitude/PN", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( meaopn02_[ism-1] ) dqmStore_->removeElement( meaopn02_[ism-1]->getName() );
      sprintf(histo, "EBLT amplitude over PN L2 %s", Numbers::sEB(ism).c_str());
      meaopn02_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      meaopn02_[ism-1]->setAxisTitle("channel", 1);
      meaopn02_[ism-1]->setAxisTitle("amplitude/PN", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( meaopn03_[ism-1] ) dqmStore_->removeElement( meaopn03_[ism-1]->getName() );
      sprintf(histo, "EBLT amplitude over PN L3 %s", Numbers::sEB(ism).c_str());
      meaopn03_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      meaopn03_[ism-1]->setAxisTitle("channel", 1);
      meaopn03_[ism-1]->setAxisTitle("amplitude/PN", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( meaopn04_[ism-1] ) dqmStore_->removeElement( meaopn04_[ism-1]->getName() );
      sprintf(histo, "EBLT amplitude over PN L4 %s", Numbers::sEB(ism).c_str());
      meaopn04_[ism-1] = dqmStore_->book1D(histo, histo, 1700, 0., 1700.);
      meaopn04_[ism-1]->setAxisTitle("channel", 1);
      meaopn04_[ism-1]->setAxisTitle("amplitude/PN", 2);
    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( mepnprms01_[ism-1] ) dqmStore_->removeElement( mepnprms01_[ism-1]->getName() );
      sprintf(histo, "EBLT PNs pedestal rms %s G01 L1", Numbers::sEB(ism).c_str());
      mepnprms01_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms01_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( mepnprms02_[ism-1] ) dqmStore_->removeElement( mepnprms02_[ism-1]->getName() );
      sprintf(histo, "EBLT PNs pedestal rms %s G01 L2", Numbers::sEB(ism).c_str());
      mepnprms02_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms02_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( mepnprms03_[ism-1] ) dqmStore_->removeElement( mepnprms03_[ism-1]->getName() );
      sprintf(histo, "EBLT PNs pedestal rms %s G01 L3", Numbers::sEB(ism).c_str());
      mepnprms03_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms03_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( mepnprms04_[ism-1] ) dqmStore_->removeElement( mepnprms04_[ism-1]->getName() );
      sprintf(histo, "EBLT PNs pedestal rms %s G01 L4", Numbers::sEB(ism).c_str());
      mepnprms04_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms04_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( mepnprms05_[ism-1] ) dqmStore_->removeElement( mepnprms05_[ism-1]->getName() );
      sprintf(histo, "EBLT PNs pedestal rms %s G16 L1", Numbers::sEB(ism).c_str());
      mepnprms05_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms05_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( mepnprms06_[ism-1] ) dqmStore_->removeElement( mepnprms06_[ism-1]->getName() );
      sprintf(histo, "EBLT PNs pedestal rms %s G16 L2", Numbers::sEB(ism).c_str());
      mepnprms06_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms06_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( mepnprms07_[ism-1] ) dqmStore_->removeElement( mepnprms07_[ism-1]->getName() );
      sprintf(histo, "EBLT PNs pedestal rms %s G16 L3", Numbers::sEB(ism).c_str());
      mepnprms07_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms07_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( mepnprms08_[ism-1] ) dqmStore_->removeElement( mepnprms08_[ism-1]->getName() );
      sprintf(histo, "EBLT PNs pedestal rms %s G16 L4", Numbers::sEB(ism).c_str());
      mepnprms08_[ism-1] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
      mepnprms08_[ism-1]->setAxisTitle("rms", 1);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
      if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
      sprintf(histo, "EBLT laser shape L1 %s", Numbers::sEB(ism).c_str());
      me_hs01_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
      me_hs01_[ism-1]->setAxisTitle("sample", 1);
      me_hs01_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
      if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
      sprintf(histo, "EBLT laser shape L2 %s", Numbers::sEB(ism).c_str());
      me_hs02_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
      me_hs02_[ism-1]->setAxisTitle("sample", 1);
      me_hs02_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
      if ( me_hs03_[ism-1] ) dqmStore_->removeElement( me_hs03_[ism-1]->getName() );
      sprintf(histo, "EBLT laser shape L3 %s", Numbers::sEB(ism).c_str());
      me_hs03_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
      me_hs03_[ism-1]->setAxisTitle("sample", 1);
      me_hs03_[ism-1]->setAxisTitle("amplitude", 2);
    }
    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
      if ( me_hs04_[ism-1] ) dqmStore_->removeElement( me_hs04_[ism-1]->getName() );
      sprintf(histo, "EBLT laser shape L4 %s", Numbers::sEB(ism).c_str());
      me_hs04_[ism-1] = dqmStore_->book1D(histo, histo, 10, 0., 10.);
      me_hs04_[ism-1]->setAxisTitle("sample", 1);
      me_hs04_[ism-1]->setAxisTitle("amplitude", 2);
    }

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();
    if ( meg04_[ism-1] ) meg04_[ism-1]->Reset();

    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();
    if ( meg06_[ism-1] ) meg06_[ism-1]->Reset();
    if ( meg07_[ism-1] ) meg07_[ism-1]->Reset();
    if ( meg08_[ism-1] ) meg08_[ism-1]->Reset();
    if ( meg09_[ism-1] ) meg09_[ism-1]->Reset();
    if ( meg10_[ism-1] ) meg10_[ism-1]->Reset();
    if ( meg11_[ism-1] ) meg11_[ism-1]->Reset();
    if ( meg12_[ism-1] ) meg12_[ism-1]->Reset();

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2. );
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2. );
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, 2. );
        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ie, ip, 2. );

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

        if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg07_[ism-1] ) meg07_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg08_[ism-1] ) meg08_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg10_[ism-1] ) meg10_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg11_[ism-1] ) meg11_[ism-1]->setBinContent( i, 1, 2. );
        if ( meg12_[ism-1] ) meg12_[ism-1]->setBinContent( i, 1, 2. );

    }

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mea02_[ism-1] ) mea02_[ism-1]->Reset();
    if ( mea03_[ism-1] ) mea03_[ism-1]->Reset();
    if ( mea04_[ism-1] ) mea04_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();
    if ( met03_[ism-1] ) met03_[ism-1]->Reset();
    if ( met04_[ism-1] ) met04_[ism-1]->Reset();

    if ( metav01_[ism-1] ) metav01_[ism-1]->Reset();
    if ( metav02_[ism-1] ) metav02_[ism-1]->Reset();
    if ( metav03_[ism-1] ) metav03_[ism-1]->Reset();
    if ( metav04_[ism-1] ) metav04_[ism-1]->Reset();

    if ( metrms01_[ism-1] ) metrms01_[ism-1]->Reset();
    if ( metrms02_[ism-1] ) metrms02_[ism-1]->Reset();
    if ( metrms03_[ism-1] ) metrms03_[ism-1]->Reset();
    if ( metrms04_[ism-1] ) metrms04_[ism-1]->Reset();

    if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->Reset();
    if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->Reset();
    if ( meaopn03_[ism-1] ) meaopn03_[ism-1]->Reset();
    if ( meaopn04_[ism-1] ) meaopn04_[ism-1]->Reset();

    if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Reset();
    if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Reset();
    if ( mepnprms03_[ism-1] ) mepnprms03_[ism-1]->Reset();
    if ( mepnprms04_[ism-1] ) mepnprms04_[ism-1]->Reset();
    if ( mepnprms05_[ism-1] ) mepnprms01_[ism-1]->Reset();
    if ( mepnprms06_[ism-1] ) mepnprms02_[ism-1]->Reset();
    if ( mepnprms07_[ism-1] ) mepnprms03_[ism-1]->Reset();
    if ( mepnprms08_[ism-1] ) mepnprms04_[ism-1]->Reset();

    if ( me_hs01_[ism-1] ) me_hs01_[ism-1]->Reset();
    if ( me_hs02_[ism-1] ) me_hs02_[ism-1]->Reset();
    if ( me_hs03_[ism-1] ) me_hs03_[ism-1]->Reset();
    if ( me_hs04_[ism-1] ) me_hs04_[ism-1]->Reset();

  }

}

void EBLaserClient::cleanup(void) {

  if ( ! enableCleanup_ ) return;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( cloneME_ ) {
      if ( h01_[ism-1] ) delete h01_[ism-1];
      if ( h02_[ism-1] ) delete h02_[ism-1];
      if ( h03_[ism-1] ) delete h03_[ism-1];
      if ( h04_[ism-1] ) delete h04_[ism-1];
      if ( h05_[ism-1] ) delete h05_[ism-1];
      if ( h06_[ism-1] ) delete h06_[ism-1];
      if ( h07_[ism-1] ) delete h07_[ism-1];
      if ( h08_[ism-1] ) delete h08_[ism-1];

      if ( h09_[ism-1] ) delete h09_[ism-1];
      if ( h10_[ism-1] ) delete h10_[ism-1];
      if ( h11_[ism-1] ) delete h11_[ism-1];
      if ( h12_[ism-1] ) delete h12_[ism-1];

      if ( hs01_[ism-1] ) delete hs01_[ism-1];
      if ( hs02_[ism-1] ) delete hs02_[ism-1];
      if ( hs03_[ism-1] ) delete hs03_[ism-1];
      if ( hs04_[ism-1] ) delete hs04_[ism-1];

      if ( i01_[ism-1] ) delete i01_[ism-1];
      if ( i02_[ism-1] ) delete i02_[ism-1];
      if ( i03_[ism-1] ) delete i03_[ism-1];
      if ( i04_[ism-1] ) delete i04_[ism-1];
      if ( i05_[ism-1] ) delete i05_[ism-1];
      if ( i06_[ism-1] ) delete i06_[ism-1];
      if ( i07_[ism-1] ) delete i07_[ism-1];
      if ( i08_[ism-1] ) delete i08_[ism-1];

      if ( i09_[ism-1] ) delete i09_[ism-1];
      if ( i10_[ism-1] ) delete i10_[ism-1];
      if ( i11_[ism-1] ) delete i11_[ism-1];
      if ( i12_[ism-1] ) delete i12_[ism-1];
      if ( i13_[ism-1] ) delete i13_[ism-1];
      if ( i14_[ism-1] ) delete i14_[ism-1];
      if ( i15_[ism-1] ) delete i15_[ism-1];
      if ( i16_[ism-1] ) delete i16_[ism-1];
    }

    h01_[ism-1] = 0;
    h02_[ism-1] = 0;
    h03_[ism-1] = 0;
    h04_[ism-1] = 0;
    h05_[ism-1] = 0;
    h06_[ism-1] = 0;
    h07_[ism-1] = 0;
    h08_[ism-1] = 0;

    h09_[ism-1] = 0;
    h10_[ism-1] = 0;
    h11_[ism-1] = 0;
    h12_[ism-1] = 0;

    hs01_[ism-1] = 0;
    hs02_[ism-1] = 0;
    hs03_[ism-1] = 0;
    hs04_[ism-1] = 0;

    i01_[ism-1] = 0;
    i02_[ism-1] = 0;
    i03_[ism-1] = 0;
    i04_[ism-1] = 0;
    i05_[ism-1] = 0;
    i06_[ism-1] = 0;
    i07_[ism-1] = 0;
    i08_[ism-1] = 0;

    i09_[ism-1] = 0;
    i10_[ism-1] = 0;
    i11_[ism-1] = 0;
    i12_[ism-1] = 0;
    i13_[ism-1] = 0;
    i14_[ism-1] = 0;
    i15_[ism-1] = 0;
    i16_[ism-1] = 0;

  }

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    dqmStore_->setCurrentFolder( prefixME_ + "/EBLaserClient" );

    if ( meg01_[ism-1] ) dqmStore_->removeElement( meg01_[ism-1]->getName() );
    meg01_[ism-1] = 0;
    if ( meg02_[ism-1] ) dqmStore_->removeElement( meg02_[ism-1]->getName() );
    meg02_[ism-1] = 0;
    if ( meg03_[ism-1] ) dqmStore_->removeElement( meg03_[ism-1]->getName() );
    meg03_[ism-1] = 0;
    if ( meg04_[ism-1] ) dqmStore_->removeElement( meg04_[ism-1]->getName() );
    meg04_[ism-1] = 0;

    if ( meg05_[ism-1] ) dqmStore_->removeElement( meg05_[ism-1]->getName() );
    meg05_[ism-1] = 0;
    if ( meg06_[ism-1] ) dqmStore_->removeElement( meg06_[ism-1]->getName() );
    meg06_[ism-1] = 0;
    if ( meg07_[ism-1] ) dqmStore_->removeElement( meg07_[ism-1]->getName() );
    meg07_[ism-1] = 0;
    if ( meg08_[ism-1] ) dqmStore_->removeElement( meg08_[ism-1]->getName() );
    meg08_[ism-1] = 0;
    if ( meg09_[ism-1] ) dqmStore_->removeElement( meg09_[ism-1]->getName() );
    meg09_[ism-1] = 0;
    if ( meg10_[ism-1] ) dqmStore_->removeElement( meg10_[ism-1]->getName() );
    meg10_[ism-1] = 0;
    if ( meg11_[ism-1] ) dqmStore_->removeElement( meg11_[ism-1]->getName() );
    meg11_[ism-1] = 0;
    if ( meg12_[ism-1] ) dqmStore_->removeElement( meg12_[ism-1]->getName() );
    meg12_[ism-1] = 0;

    if ( mea01_[ism-1] ) dqmStore_->removeElement( mea01_[ism-1]->getName() );
    mea01_[ism-1] = 0;
    if ( mea02_[ism-1] ) dqmStore_->removeElement( mea02_[ism-1]->getName() );
    mea02_[ism-1] = 0;
    if ( mea03_[ism-1] ) dqmStore_->removeElement( mea03_[ism-1]->getName() );
    mea03_[ism-1] = 0;
    if ( mea04_[ism-1] ) dqmStore_->removeElement( mea04_[ism-1]->getName() );
    mea04_[ism-1] = 0;

    if ( met01_[ism-1] ) dqmStore_->removeElement( met01_[ism-1]->getName() );
    met01_[ism-1] = 0;
    if ( met02_[ism-1] ) dqmStore_->removeElement( met02_[ism-1]->getName() );
    met02_[ism-1] = 0;
    if ( met03_[ism-1] ) dqmStore_->removeElement( met03_[ism-1]->getName() );
    met03_[ism-1] = 0;
    if ( met04_[ism-1] ) dqmStore_->removeElement( met04_[ism-1]->getName() );
    met04_[ism-1] = 0;

    if ( metav01_[ism-1] ) dqmStore_->removeElement( metav01_[ism-1]->getName() );
    metav01_[ism-1] = 0;
    if ( metav02_[ism-1] ) dqmStore_->removeElement( metav02_[ism-1]->getName() );
    metav02_[ism-1] = 0;
    if ( metav03_[ism-1] ) dqmStore_->removeElement( metav03_[ism-1]->getName() );
    metav03_[ism-1] = 0;
    if ( metav04_[ism-1] ) dqmStore_->removeElement( metav04_[ism-1]->getName() );
    metav04_[ism-1] = 0;

    if ( metrms01_[ism-1] ) dqmStore_->removeElement( metrms01_[ism-1]->getName() );
    metrms01_[ism-1] = 0;
    if ( metrms02_[ism-1] ) dqmStore_->removeElement( metrms02_[ism-1]->getName() );
    metrms02_[ism-1] = 0;
    if ( metrms03_[ism-1] ) dqmStore_->removeElement( metrms03_[ism-1]->getName() );
    metrms03_[ism-1] = 0;
    if ( metrms04_[ism-1] ) dqmStore_->removeElement( metrms04_[ism-1]->getName() );
    metrms04_[ism-1] = 0;

    if ( meaopn01_[ism-1] ) dqmStore_->removeElement( meaopn01_[ism-1]->getName() );
    meaopn01_[ism-1] = 0;
    if ( meaopn02_[ism-1] ) dqmStore_->removeElement( meaopn02_[ism-1]->getName() );
    meaopn02_[ism-1] = 0;
    if ( meaopn03_[ism-1] ) dqmStore_->removeElement( meaopn03_[ism-1]->getName() );
    meaopn03_[ism-1] = 0;
    if ( meaopn04_[ism-1] ) dqmStore_->removeElement( meaopn04_[ism-1]->getName() );
    meaopn04_[ism-1] = 0;

    if ( mepnprms01_[ism-1] ) dqmStore_->removeElement( mepnprms01_[ism-1]->getName() );
    mepnprms01_[ism-1] = 0;
    if ( mepnprms02_[ism-1] ) dqmStore_->removeElement( mepnprms02_[ism-1]->getName() );
    mepnprms02_[ism-1] = 0;
    if ( mepnprms03_[ism-1] ) dqmStore_->removeElement( mepnprms03_[ism-1]->getName() );
    mepnprms03_[ism-1] = 0;
    if ( mepnprms04_[ism-1] ) dqmStore_->removeElement( mepnprms04_[ism-1]->getName() );
    mepnprms04_[ism-1] = 0;
    if ( mepnprms05_[ism-1] ) dqmStore_->removeElement( mepnprms05_[ism-1]->getName() );
    mepnprms05_[ism-1] = 0;
    if ( mepnprms06_[ism-1] ) dqmStore_->removeElement( mepnprms06_[ism-1]->getName() );
    mepnprms06_[ism-1] = 0;
    if ( mepnprms07_[ism-1] ) dqmStore_->removeElement( mepnprms07_[ism-1]->getName() );
    mepnprms07_[ism-1] = 0;
    if ( mepnprms08_[ism-1] ) dqmStore_->removeElement( mepnprms08_[ism-1]->getName() );
    mepnprms08_[ism-1] = 0;

    if ( me_hs01_[ism-1] ) dqmStore_->removeElement( me_hs01_[ism-1]->getName() );
    me_hs01_[ism-1] = 0;
    if ( me_hs02_[ism-1] ) dqmStore_->removeElement( me_hs02_[ism-1]->getName() );
    me_hs02_[ism-1] = 0;
    if ( me_hs03_[ism-1] ) dqmStore_->removeElement( me_hs03_[ism-1]->getName() );
    me_hs03_[ism-1] = 0;
    if ( me_hs04_[ism-1] ) dqmStore_->removeElement( me_hs04_[ism-1]->getName() );
    me_hs04_[ism-1] = 0;

  }

}

bool EBLaserClient::writeDb(EcalCondDBInterface* econn, RunIOV* runiov, MonRunIOV* moniov, bool& status, bool flag) {

  status = true;

  if ( ! flag ) return false;

  EcalLogicID ecid;

  MonLaserBlueDat apd_bl;
  map<EcalLogicID, MonLaserBlueDat> dataset1_bl;
  MonLaserGreenDat apd_gr;
  map<EcalLogicID, MonLaserGreenDat> dataset1_gr;
  MonLaserIRedDat apd_ir;
  map<EcalLogicID, MonLaserIRedDat> dataset1_ir;
  MonLaserRedDat apd_rd;
  map<EcalLogicID, MonLaserRedDat> dataset1_rd;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg01_[ism-1], h01_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg02_[ism-1], h03_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg03_[ism-1], h05_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg04_[ism-1], h07_[ism-1]);
      }
    }

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update01;
        bool update02;
        bool update03;
        bool update04;
        bool update05;
        bool update06;
        bool update07;
        bool update08;

        float num01, num02, num03, num04, num05, num06, num07, num08;
        float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
        float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ie, ip, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h04_[ism-1], ie, ip, num04, mean04, rms04);
        update05 = UtilsClient::getBinStatistics(h05_[ism-1], ie, ip, num05, mean05, rms05);
        update06 = UtilsClient::getBinStatistics(h06_[ism-1], ie, ip, num06, mean06, rms06);
        update07 = UtilsClient::getBinStatistics(h07_[ism-1], ie, ip, num07, mean07, rms07);
        update08 = UtilsClient::getBinStatistics(h08_[ism-1], ie, ip, num08, mean08, rms08);

        if ( update01 || update02 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1 (" << ie << "," << ip << ") " << num01 << " " << mean01 << " " << rms01 << endl;
              cout << endl;
            }

          }

          apd_bl.setAPDMean(mean01);
          apd_bl.setAPDRMS(rms01);

          apd_bl.setAPDOverPNMean(mean02);
          apd_bl.setAPDOverPNRMS(rms02);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ie, ip) ) {
            apd_bl.setTaskStatus(true);
          } else {
            apd_bl.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset1_bl[ecid] = apd_bl;
          }

        }

        if ( update03 || update04 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "L2 (" << ie << "," << ip << ") " << num03 << " " << mean03 << " " << rms03 << endl;
              cout << endl;
            }

          }

          apd_ir.setAPDMean(mean03);
          apd_ir.setAPDRMS(rms03);

          apd_ir.setAPDOverPNMean(mean04);
          apd_ir.setAPDOverPNRMS(rms04);

          if ( UtilsClient::getBinStatus(meg02_[ism-1], ie, ip) ) {
            apd_ir.setTaskStatus(true);
          } else {
            apd_ir.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg02_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset1_ir[ecid] = apd_ir;
          }

        }

        if ( update05 || update06 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "L3 (" << ie << "," << ip << ") " << num05 << " " << mean05 << " " << rms05 << endl;
              cout << endl;
            }

          }

          apd_gr.setAPDMean(mean05);
          apd_gr.setAPDRMS(rms05);

          apd_gr.setAPDOverPNMean(mean06);
          apd_gr.setAPDOverPNRMS(rms06);

          if ( UtilsClient::getBinStatus(meg03_[ism-1], ie, ip) ) {
            apd_gr.setTaskStatus(true);
          } else {
            apd_gr.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg03_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset1_gr[ecid] = apd_gr;
          }

        }

        if ( update07 || update08 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "L4 (" << ie << "," << ip << ") " << num07 << " " << mean07 << " " << rms07 << endl;
              cout << endl;
            }

          }

          apd_rd.setAPDMean(mean07);
          apd_rd.setAPDRMS(rms07);

          apd_rd.setAPDOverPNMean(mean08);
          apd_rd.setAPDOverPNRMS(rms08);

          if ( UtilsClient::getBinStatus(meg04_[ism-1], ie, ip) ) {
            apd_rd.setTaskStatus(true);
          } else {
            apd_rd.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg04_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset1_rd[ecid] = apd_rd;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonLaserDat ..." << endl;
      if ( dataset1_bl.size() != 0 ) econn->insertDataArraySet(&dataset1_bl, moniov);
      if ( dataset1_ir.size() != 0 ) econn->insertDataArraySet(&dataset1_ir, moniov);
      if ( dataset1_gr.size() != 0 ) econn->insertDataArraySet(&dataset1_gr, moniov);
      if ( dataset1_rd.size() != 0 ) econn->insertDataArraySet(&dataset1_rd, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( verbose_ ) cout << endl;

  MonPNBlueDat pn_bl;
  map<EcalLogicID, MonPNBlueDat> dataset2_bl;
  MonPNGreenDat pn_gr;
  map<EcalLogicID, MonPNGreenDat> dataset2_gr;
  MonPNIRedDat pn_ir;
  map<EcalLogicID, MonPNIRedDat> dataset2_ir;
  MonPNRedDat pn_rd;
  map<EcalLogicID, MonPNRedDat> dataset2_rd;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg05_[ism-1], i01_[ism-1]);
        UtilsClient::printBadChannels(meg05_[ism-1], i05_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg06_[ism-1], i02_[ism-1]);
        UtilsClient::printBadChannels(meg06_[ism-1], i06_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg07_[ism-1], i03_[ism-1]);
        UtilsClient::printBadChannels(meg07_[ism-1], i07_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg08_[ism-1], i04_[ism-1]);
        UtilsClient::printBadChannels(meg08_[ism-1], i08_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg09_[ism-1], i09_[ism-1]);
        UtilsClient::printBadChannels(meg09_[ism-1], i13_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg10_[ism-1], i10_[ism-1]);
        UtilsClient::printBadChannels(meg10_[ism-1], i14_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg11_[ism-1], i11_[ism-1]);
        UtilsClient::printBadChannels(meg11_[ism-1], i15_[ism-1]);
      }
      if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {
        UtilsClient::printBadChannels(meg12_[ism-1], i12_[ism-1]);
        UtilsClient::printBadChannels(meg12_[ism-1], i16_[ism-1]);
      }
    }

    for ( int i = 1; i <= 10; i++ ) {

      bool update01;
      bool update02;
      bool update03;
      bool update04;
      bool update05;
      bool update06;
      bool update07;
      bool update08;
      bool update09;
      bool update10;
      bool update11;
      bool update12;
      bool update13;
      bool update14;
      bool update15;
      bool update16;

      float num01, num02, num03, num04, num05, num06, num07, num08;
      float num09, num10, num11, num12, num13, num14, num15, num16;
      float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
      float mean09, mean10, mean11, mean12, mean13, mean14, mean15, mean16;
      float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
      float rms09, rms10, rms11, rms12, rms13, rms14, rms15, rms16;

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);
      update03 = UtilsClient::getBinStatistics(i03_[ism-1], i, 0, num03, mean03, rms03);
      update04 = UtilsClient::getBinStatistics(i04_[ism-1], i, 0, num04, mean04, rms04);
      update05 = UtilsClient::getBinStatistics(i05_[ism-1], i, 0, num05, mean05, rms05);
      update06 = UtilsClient::getBinStatistics(i06_[ism-1], i, 0, num06, mean06, rms06);
      update07 = UtilsClient::getBinStatistics(i07_[ism-1], i, 0, num07, mean07, rms07);
      update08 = UtilsClient::getBinStatistics(i08_[ism-1], i, 0, num08, mean08, rms08);
      update09 = UtilsClient::getBinStatistics(i09_[ism-1], i, 0, num09, mean09, rms09);
      update10 = UtilsClient::getBinStatistics(i10_[ism-1], i, 0, num10, mean10, rms10);
      update11 = UtilsClient::getBinStatistics(i11_[ism-1], i, 0, num11, mean11, rms11);
      update12 = UtilsClient::getBinStatistics(i12_[ism-1], i, 0, num12, mean12, rms12);
      update13 = UtilsClient::getBinStatistics(i13_[ism-1], i, 0, num13, mean13, rms13);
      update14 = UtilsClient::getBinStatistics(i14_[ism-1], i, 0, num14, mean14, rms14);
      update15 = UtilsClient::getBinStatistics(i15_[ism-1], i, 0, num15, mean15, rms15);
      update16 = UtilsClient::getBinStatistics(i16_[ism-1], i, 0, num16, mean16, rms16);

      if ( update01 || update05 || update09 || update13 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") L1 G01 " << num01  << " " << mean01 << " " << rms01  << endl;
            cout << "PNs (" << i << ") L1 G16 " << num09  << " " << mean09 << " " << rms09  << endl;
            cout << endl;
          }

        }

        pn_bl.setADCMeanG1(mean01);
        pn_bl.setADCRMSG1(rms01);

        pn_bl.setPedMeanG1(mean05);
        pn_bl.setPedRMSG1(rms05);

        pn_bl.setADCMeanG16(mean09);
        pn_bl.setADCRMSG16(rms09);

        pn_bl.setPedMeanG16(mean13);
        pn_bl.setPedRMSG16(rms13);

        if ( UtilsClient::getBinStatus(meg05_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg09_[ism-1], i, 1) ) {
          pn_bl.setTaskStatus(true);
        } else {
          pn_bl.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg05_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg09_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism, EcalBarrel), i-1);
          dataset2_bl[ecid] = pn_bl;
        }

      }

      if ( update02 || update06 || update10 || update14 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") L2 G01 " << num02  << " " << mean02 << " " << rms02  << endl;
            cout << "PNs (" << i << ") L2 G16 " << num10  << " " << mean10 << " " << rms10  << endl;
            cout << endl;
          }

        }

        pn_ir.setADCMeanG1(mean02);
        pn_ir.setADCRMSG1(rms02);

        pn_ir.setPedMeanG1(mean06);
        pn_ir.setPedRMSG1(rms06);

        pn_ir.setADCMeanG16(mean10);
        pn_ir.setADCRMSG16(rms10);

        pn_ir.setPedMeanG16(mean14);
        pn_ir.setPedRMSG16(rms14);

        if ( UtilsClient::getBinStatus(meg06_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg10_[ism-1], i, 1) ) {
          pn_ir.setTaskStatus(true);
        } else {
          pn_ir.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg06_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg10_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism, EcalBarrel), i-1);
          dataset2_ir[ecid] = pn_ir;
        }

      }

      if ( update03 || update07 || update11 || update15 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") L3 G01 " << num03  << " " << mean03 << " " << rms03  << endl;
            cout << "PNs (" << i << ") L3 G16 " << num11  << " " << mean11 << " " << rms11  << endl;
            cout << endl;
          }

        }

        pn_gr.setADCMeanG1(mean03);
        pn_gr.setADCRMSG1(rms03);

        pn_gr.setPedMeanG1(mean07);
        pn_gr.setPedRMSG1(rms07);

        pn_gr.setADCMeanG16(mean11);
        pn_gr.setADCRMSG16(rms11);

        pn_gr.setPedMeanG16(mean15);
        pn_gr.setPedRMSG16(rms15);

        if ( UtilsClient::getBinStatus(meg07_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg11_[ism-1], i, 1) ) {
          pn_gr.setTaskStatus(true);
        } else {
          pn_gr.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg07_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg11_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism, EcalBarrel), i-1);
          dataset2_gr[ecid] = pn_gr;
        }

      }

      if ( update04 || update08 || update12 || update16 ) {

        if ( i == 1 ) {

          if ( verbose_ ) {
            cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
            cout << "PNs (" << i << ") L4 G01 " << num04  << " " << mean04 << " " << rms04  << endl;
            cout << "PNs (" << i << ") L4 G16 " << num12  << " " << mean12 << " " << rms12  << endl;
            cout << endl;
          }

        }

        pn_rd.setADCMeanG1(mean04);
        pn_rd.setADCRMSG1(rms04);

        pn_rd.setPedMeanG1(mean08);
        pn_rd.setPedRMSG1(mean08);

        pn_rd.setADCMeanG16(mean12);
        pn_rd.setADCRMSG16(rms12);

        pn_rd.setPedMeanG16(mean16);
        pn_rd.setPedRMSG16(rms16);

        if ( UtilsClient::getBinStatus(meg08_[ism-1], i, 1) ||
             UtilsClient::getBinStatus(meg12_[ism-1], i, 1) ) {
          pn_rd.setTaskStatus(true);
        } else {
          pn_rd.setTaskStatus(false);
        }

        status = status && ( UtilsClient::getBinQuality(meg08_[ism-1], i, 1) ||
                             UtilsClient::getBinQuality(meg12_[ism-1], i, 1) );

        if ( econn ) {
          ecid = LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism, EcalBarrel), i-1);
          dataset2_rd[ecid] = pn_rd;
        }

      }

    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonPnDat ..." << endl;
      if ( dataset2_bl.size() != 0 ) econn->insertDataArraySet(&dataset2_bl, moniov);
      if ( dataset2_ir.size() != 0 ) econn->insertDataArraySet(&dataset2_ir, moniov);
      if ( dataset2_gr.size() != 0 ) econn->insertDataArraySet(&dataset2_gr, moniov);
      if ( dataset2_rd.size() != 0 ) econn->insertDataArraySet(&dataset2_rd, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  if ( verbose_ ) cout << endl;

  MonTimingLaserBlueCrystalDat t_bl;
  map<EcalLogicID, MonTimingLaserBlueCrystalDat> dataset3_bl;
  MonTimingLaserGreenCrystalDat t_gr;
  map<EcalLogicID, MonTimingLaserGreenCrystalDat> dataset3_gr;
  MonTimingLaserIRedCrystalDat t_ir;
  map<EcalLogicID, MonTimingLaserIRedCrystalDat> dataset3_ir;
  MonTimingLaserRedCrystalDat t_rd;
  map<EcalLogicID, MonTimingLaserRedCrystalDat> dataset3_rd;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( verbose_ ) {
      cout << " " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
      cout << endl;
    }

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update01;
        bool update02;
        bool update03;
        bool update04;

        float num01, num02, num03, num04;
        float mean01, mean02, mean03, mean04;
        float rms01, rms02, rms03, rms04;

        update01 = UtilsClient::getBinStatistics(h09_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h10_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h11_[ism-1], ie, ip, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h12_[ism-1], ie, ip, num04, mean04, rms04);

        if ( update01 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "L1 crystal (" << ie << "," << ip << ") " << num01  << " " << mean01 << " " << rms01  << endl;
              cout << endl;
            }

          }

          t_bl.setTimingMean(mean01);
          t_bl.setTimingRMS(rms01);

          if ( UtilsClient::getBinStatus(meg01_[ism-1], ie, ip) ) {
            t_bl.setTaskStatus(true);
          } else {
            t_bl.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg01_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset3_bl[ecid] = t_bl;
          }

        }

        if ( update02 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "L2 crystal (" << ie << "," << ip << ") " << num02  << " " << mean02 << " " << rms02  << endl;
              cout << endl;
            }

          }

          t_gr.setTimingMean(mean02);
          t_gr.setTimingRMS(rms02);

          if ( UtilsClient::getBinStatus(meg02_[ism-1], ie, ip) ) {
            t_gr.setTaskStatus(true);
          } else {
            t_gr.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg02_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset3_gr[ecid] = t_gr;
          }

        }

        if ( update03 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "L3 crystal (" << ie << "," << ip << ") " << num03  << " " << mean03 << " " << rms03  << endl;
              cout << endl;
            }

          }

          t_ir.setTimingMean(mean03);
          t_ir.setTimingRMS(rms03);

          if ( UtilsClient::getBinStatus(meg03_[ism-1], ie, ip) ) {
            t_ir.setTaskStatus(true);
          } else {
            t_ir.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg03_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset3_ir[ecid] = t_ir;
          }

        }

        if ( update04 ) {

          if ( Numbers::icEB(ism, ie, ip) == 1 ) {

            if ( verbose_ ) {
              cout << "Preparing dataset for " << Numbers::sEB(ism) << " (ism=" << ism << ")" << endl;
              cout << "L4 crystal (" << ie << "," << ip << ") " << num04  << " " << mean04 << " " << rms04  << endl;
              cout << endl;
            }

          }

          t_rd.setTimingMean(mean04);
          t_rd.setTimingRMS(rms04);

          if ( UtilsClient::getBinStatus(meg04_[ism-1], ie, ip) ) {
            t_rd.setTaskStatus(true);
          } else {
            t_rd.setTaskStatus(false);
          }

          status = status && UtilsClient::getBinQuality(meg04_[ism-1], ie, ip);

          int ic = Numbers::indexEB(ism, ie, ip);

          if ( econn ) {
            ecid = LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic);
            dataset3_rd[ecid] = t_rd;
          }

        }

      }
    }

  }

  if ( econn ) {
    try {
      if ( verbose_ ) cout << "Inserting MonTimingLaserCrystalDat ..." << endl;
      if ( dataset3_bl.size() != 0 ) econn->insertDataArraySet(&dataset3_bl, moniov);
      if ( dataset3_ir.size() != 0 ) econn->insertDataArraySet(&dataset3_ir, moniov);
      if ( dataset3_gr.size() != 0 ) econn->insertDataArraySet(&dataset3_gr, moniov);
      if ( dataset3_rd.size() != 0 ) econn->insertDataArraySet(&dataset3_rd, moniov);
      if ( verbose_ ) cout << "done." << endl;
    } catch (runtime_error &e) {
      cerr << e.what() << endl;
    }
  }

  return true;

}

void EBLaserClient::analyze(void) {

  ievt_++;
  jevt_++;
  if ( ievt_ % 10 == 0 ) {
    if ( debug_ ) cout << "EBLaserClient: ievt/jevt = " << ievt_ << "/" << jevt_ << endl;
  }

  uint64_t bits01 = 0;
  bits01 |= EcalErrorDictionary::getMask("LASER_MEAN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LASER_RMS_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LASER_MEAN_OVER_PN_WARNING");
  bits01 |= EcalErrorDictionary::getMask("LASER_RMS_OVER_PN_WARNING");

  uint64_t bits02 = 0;
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_WARNING");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_MEAN_ERROR");
  bits02 |= EcalErrorDictionary::getMask("PEDESTAL_LOW_GAIN_RMS_ERROR");

  uint64_t bits03 = 0;
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_WARNING");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_MEAN_ERROR");
  bits03 |= EcalErrorDictionary::getMask("PEDESTAL_MIDDLE_GAIN_RMS_ERROR");

  uint64_t bits04 = 0;
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_WARNING");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_WARNING");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_MEAN_ERROR");
  bits04 |= EcalErrorDictionary::getMask("PEDESTAL_HIGH_GAIN_RMS_ERROR");

  map<EcalLogicID, RunCrystalErrorsDat> mask1;
  map<EcalLogicID, RunPNErrorsDat> mask2;
  map<EcalLogicID, RunTTErrorsDat> mask3;

  EcalErrorMask::fetchDataSet(&mask1);
  EcalErrorMask::fetchDataSet(&mask2);
  EcalErrorMask::fetchDataSet(&mask3);

  char histo[200];

  MonitorElement* me;

  for ( unsigned int i=0; i<superModules_.size(); i++ ) {

    int ism = superModules_[i];

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser1/EBLT amplitude %s L1").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h01_[ism-1] );

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser1/EBLT amplitude over PN %s L1").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h02_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser2/EBLT amplitude %s L2").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h03_[ism-1] );

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser2/EBLT amplitude over PN %s L2").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h04_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h04_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser3/EBLT amplitude %s L3").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h05_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h05_[ism-1] );

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser3/EBLT amplitude over PN %s L3").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h06_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h06_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser4/EBLT amplitude %s L4").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h07_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h07_[ism-1] );

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser4/EBLT amplitude over PN %s L4").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h08_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h08_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser1/EBLT timing %s L1").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h09_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h09_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser2/EBLT timing %s L2").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h10_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h10_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser3/EBLT timing %s L3").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h11_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h11_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser4/EBLT timing %s L4").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      h12_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, h12_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser1/EBLT shape %s L1").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      hs01_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs01_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser2/EBLT shape %s L2").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      hs02_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs02_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser3/EBLT shape %s L3").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      hs03_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs03_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser4/EBLT shape %s L4").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      hs04_[ism-1] = UtilsClient::getHisto<TProfile2D*>( me, cloneME_, hs04_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser1/PN/Gain01/EBLT PNs amplitude %s G01 L1").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i01_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i01_[ism-1] );

    }


    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser2/PN/Gain01/EBLT PNs amplitude %s G01 L2").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i02_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i02_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser3/PN/Gain01/EBLT PNs amplitude %s G01 L3").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i03_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i03_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser4/PN/Gain01/EBLT PNs amplitude %s G01 L4").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i04_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i04_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser1/PN/Gain01/EBLT PNs pedestal %s G01 L1").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i05_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i05_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser2/PN/Gain01/EBLT PNs pedestal %s G01 L2").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i06_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i06_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser3/PN/Gain01/EBLT PNs pedestal %s G01 L3").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i07_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i07_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser4/PN/Gain01/EBLT PNs pedestal %s G01 L4").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i08_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i08_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser1/PN/Gain16/EBLT PNs amplitude %s G16 L1").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i09_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i09_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser2/PN/Gain16/EBLT PNs amplitude %s G16 L2").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i10_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i10_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser3/PN/Gain16/EBLT PNs amplitude %s G16 L3").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i11_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i11_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser4/PN/Gain16/EBLT PNs amplitude %s G16 L4").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i12_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i12_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 1) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser1/PN/Gain16/EBLT PNs pedestal %s G16 L1").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i13_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i13_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 2) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser2/PN/Gain16/EBLT PNs pedestal %s G16 L2").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i14_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i14_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 3) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser3/PN/Gain16/EBLT PNs pedestal %s G16 L3").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i15_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i15_[ism-1] );

    }

    if ( find(laserWavelengths_.begin(), laserWavelengths_.end(), 4) != laserWavelengths_.end() ) {

      sprintf(histo, (prefixME_ + "/EBLaserTask/Laser4/PN/Gain16/EBLT PNs pedestal %s G16 L4").c_str(), Numbers::sEB(ism).c_str());
      me = dqmStore_->get(histo);
      i16_[ism-1] = UtilsClient::getHisto<TProfile*>( me, cloneME_, i16_[ism-1] );

    }

    if ( meg01_[ism-1] ) meg01_[ism-1]->Reset();
    if ( meg02_[ism-1] ) meg02_[ism-1]->Reset();
    if ( meg03_[ism-1] ) meg03_[ism-1]->Reset();
    if ( meg04_[ism-1] ) meg04_[ism-1]->Reset();

    if ( meg05_[ism-1] ) meg05_[ism-1]->Reset();
    if ( meg06_[ism-1] ) meg06_[ism-1]->Reset();
    if ( meg07_[ism-1] ) meg07_[ism-1]->Reset();
    if ( meg08_[ism-1] ) meg08_[ism-1]->Reset();
    if ( meg09_[ism-1] ) meg09_[ism-1]->Reset();
    if ( meg10_[ism-1] ) meg10_[ism-1]->Reset();
    if ( meg11_[ism-1] ) meg11_[ism-1]->Reset();
    if ( meg12_[ism-1] ) meg12_[ism-1]->Reset();

    if ( mea01_[ism-1] ) mea01_[ism-1]->Reset();
    if ( mea02_[ism-1] ) mea02_[ism-1]->Reset();
    if ( mea03_[ism-1] ) mea03_[ism-1]->Reset();
    if ( mea04_[ism-1] ) mea04_[ism-1]->Reset();

    if ( met01_[ism-1] ) met01_[ism-1]->Reset();
    if ( met02_[ism-1] ) met02_[ism-1]->Reset();
    if ( met03_[ism-1] ) met03_[ism-1]->Reset();
    if ( met04_[ism-1] ) met04_[ism-1]->Reset();

    if ( metav01_[ism-1] ) metav01_[ism-1]->Reset();
    if ( metav02_[ism-1] ) metav02_[ism-1]->Reset();
    if ( metav03_[ism-1] ) metav03_[ism-1]->Reset();
    if ( metav04_[ism-1] ) metav04_[ism-1]->Reset();

    if ( metrms01_[ism-1] ) metrms01_[ism-1]->Reset();
    if ( metrms02_[ism-1] ) metrms02_[ism-1]->Reset();
    if ( metrms03_[ism-1] ) metrms03_[ism-1]->Reset();
    if ( metrms04_[ism-1] ) metrms04_[ism-1]->Reset();

    if ( meaopn01_[ism-1] ) meaopn01_[ism-1]->Reset();
    if ( meaopn02_[ism-1] ) meaopn02_[ism-1]->Reset();
    if ( meaopn03_[ism-1] ) meaopn03_[ism-1]->Reset();
    if ( meaopn04_[ism-1] ) meaopn04_[ism-1]->Reset();

    if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Reset();
    if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Reset();
    if ( mepnprms03_[ism-1] ) mepnprms03_[ism-1]->Reset();
    if ( mepnprms04_[ism-1] ) mepnprms04_[ism-1]->Reset();
    if ( mepnprms05_[ism-1] ) mepnprms05_[ism-1]->Reset();
    if ( mepnprms06_[ism-1] ) mepnprms06_[ism-1]->Reset();
    if ( mepnprms07_[ism-1] ) mepnprms07_[ism-1]->Reset();
    if ( mepnprms08_[ism-1] ) mepnprms08_[ism-1]->Reset();

    if ( me_hs01_[ism-1] ) me_hs01_[ism-1]->Reset();
    if ( me_hs02_[ism-1] ) me_hs02_[ism-1]->Reset();
    if ( me_hs03_[ism-1] ) me_hs03_[ism-1]->Reset();
    if ( me_hs04_[ism-1] ) me_hs04_[ism-1]->Reset();

    float meanAmplL1, meanAmplL2, meanAmplL3, meanAmplL4;

    int nCryL1, nCryL2, nCryL3, nCryL4;

    meanAmplL1 = meanAmplL2 = meanAmplL3 = meanAmplL4 = 0.;

    nCryL1 = nCryL2 = nCryL3 = nCryL4 = 0;

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        bool update01;
        bool update02;
        bool update03;
        bool update04;

        float num01, num02, num03, num04;
        float mean01, mean02, mean03, mean04;
        float rms01, rms02, rms03, rms04;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h03_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h05_[ism-1], ie, ip, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h07_[ism-1], ie, ip, num04, mean04, rms04);

        if ( update01 ) {
          meanAmplL1 += mean01;
          nCryL1++;
        }

        if ( update02 ) {
          meanAmplL2 += mean02;
          nCryL2++;
        }

        if ( update03 ) {
          meanAmplL3 += mean03;
          nCryL3++;
        }

        if ( update04 ) {
          meanAmplL4 += mean04;
          nCryL4++;
        }

      }
    }

    if ( nCryL1 > 0 ) meanAmplL1 /= float (nCryL1);
    if ( nCryL2 > 0 ) meanAmplL2 /= float (nCryL2);
    if ( nCryL3 > 0 ) meanAmplL3 /= float (nCryL3);
    if ( nCryL4 > 0 ) meanAmplL4 /= float (nCryL4);

    for ( int ie = 1; ie <= 85; ie++ ) {
      for ( int ip = 1; ip <= 20; ip++ ) {

        if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, 2.);
        if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, 2.);
        if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, 2.);
        if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ie, ip, 2.);

        bool update01;
        bool update02;
        bool update03;
        bool update04;
        bool update05;
        bool update06;
        bool update07;
        bool update08;
        bool update09;
        bool update10;
        bool update11;
        bool update12;

        float num01, num02, num03, num04, num05, num06, num07, num08;
        float num09, num10, num11, num12;
        float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
        float mean09, mean10, mean11, mean12;
        float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
        float rms09, rms10, rms11, rms12;

        update01 = UtilsClient::getBinStatistics(h01_[ism-1], ie, ip, num01, mean01, rms01);
        update02 = UtilsClient::getBinStatistics(h02_[ism-1], ie, ip, num02, mean02, rms02);
        update03 = UtilsClient::getBinStatistics(h03_[ism-1], ie, ip, num03, mean03, rms03);
        update04 = UtilsClient::getBinStatistics(h04_[ism-1], ie, ip, num04, mean04, rms04);
        update05 = UtilsClient::getBinStatistics(h05_[ism-1], ie, ip, num05, mean05, rms05);
        update06 = UtilsClient::getBinStatistics(h06_[ism-1], ie, ip, num06, mean06, rms06);
        update07 = UtilsClient::getBinStatistics(h07_[ism-1], ie, ip, num07, mean07, rms07);
        update08 = UtilsClient::getBinStatistics(h08_[ism-1], ie, ip, num08, mean08, rms08);
        update09 = UtilsClient::getBinStatistics(h09_[ism-1], ie, ip, num09, mean09, rms09);
        update10 = UtilsClient::getBinStatistics(h10_[ism-1], ie, ip, num10, mean10, rms10);
        update11 = UtilsClient::getBinStatistics(h11_[ism-1], ie, ip, num11, mean11, rms11);
        update12 = UtilsClient::getBinStatistics(h12_[ism-1], ie, ip, num12, mean12, rms12);

        if ( update01 ) {

          float val;

          val = 1.;
          if ( fabs(mean01 - meanAmplL1) > fabs(percentVariation_ * meanAmplL1) || mean01 < amplitudeThreshold_ || rms01 > rmsThresholdRelative_ * mean01 )
            val = 0.;
          if ( meg01_[ism-1] ) meg01_[ism-1]->setBinContent( ie, ip, val );

          int ic = Numbers::icEB(ism, ie, ip);

          if ( mea01_[ism-1] ) {
            if ( mean01 > 0. ) {
              mea01_[ism-1]->setBinContent( ic, mean01 );
              mea01_[ism-1]->setBinError( ic, rms01 );
            } else {
              mea01_[ism-1]->setEntries( 1.+mea01_[ism-1]->getEntries() );
            }
          }

        }

        if ( update03 ) {

          float val;

          val = 1.;
          if ( fabs(mean03 - meanAmplL2) > fabs(percentVariation_ * meanAmplL2) || mean03 < amplitudeThreshold_ || rms03 > rmsThresholdRelative_ * mean03 )
            val = 0.;
          if ( meg02_[ism-1] ) meg02_[ism-1]->setBinContent( ie, ip, val);

          int ic = Numbers::icEB(ism, ie, ip);

          if ( mea02_[ism-1] ) {
            if ( mean03 > 0. ) {
              mea02_[ism-1]->setBinContent( ic, mean03 );
              mea02_[ism-1]->setBinError( ic, rms03 );
            } else {
              mea02_[ism-1]->setEntries( 1.+mea02_[ism-1]->getEntries() );
            }
          }

        }

        if ( update05 ) {

          float val;

          val = 1.;
          if ( fabs(mean05 - meanAmplL3) > fabs(percentVariation_ * meanAmplL3) || mean05 < amplitudeThreshold_ || rms05 > rmsThresholdRelative_ * mean05 )
            val = 0.;
          if ( meg03_[ism-1] ) meg03_[ism-1]->setBinContent( ie, ip, val );

          int ic = Numbers::icEB(ism, ie, ip);

          if ( mea03_[ism-1] ) {
            if ( mean05 > 0. ) {
              mea03_[ism-1]->setBinContent( ic, mean05 );
              mea03_[ism-1]->setBinError( ic, rms05 );
            } else {
              mea03_[ism-1]->setEntries( 1.+mea03_[ism-1]->getEntries() );
            }
          }

        }

        if ( update07 ) {

          float val;

          val = 1.;
          if ( fabs(mean07 - meanAmplL4) > fabs(percentVariation_ * meanAmplL4) || mean07 < amplitudeThreshold_ || rms07 > rmsThresholdRelative_ * mean07 )
            val = 0.;
          if ( meg04_[ism-1] ) meg04_[ism-1]->setBinContent( ie, ip, val );

          int ic = Numbers::icEB(ism, ie, ip);

          if ( mea04_[ism-1] ) {
            if ( mean07 > 0. ) {
              mea04_[ism-1]->setBinContent( ic, mean07 );
              mea04_[ism-1]->setBinError( ic, rms07 );
            } else {
              mea04_[ism-1]->setEntries( 1.+mea04_[ism-1]->getEntries() );
            }
          }

        }

        if ( update02 ) {

          int ic = Numbers::icEB(ism, ie, ip);

          if ( meaopn01_[ism-1] ) {
            if ( mean02 > 0. ) {
              meaopn01_[ism-1]->setBinContent( ic, mean02 );
              meaopn01_[ism-1]->setBinError( ic, rms02 );
            } else {
              meaopn01_[ism-1]->setEntries( 1.+meaopn01_[ism-1]->getEntries() );
            }
          }

        }

        if ( update04 ) {

          int ic = Numbers::icEB(ism, ie, ip);

          if ( meaopn02_[ism-1] ) {
            if ( mean04 > 0. ) {
              meaopn02_[ism-1]->setBinContent( ic, mean04 );
              meaopn02_[ism-1]->setBinError( ic, rms04 );
            } else {
              meaopn02_[ism-1]->setEntries( 1.+meaopn02_[ism-1]->getEntries() );
            }
          }

        }

        if ( update06 ) {

          int ic = Numbers::icEB(ism, ie, ip);

          if ( meaopn03_[ism-1] ) {
            if ( mean06 > 0. ) {
              meaopn03_[ism-1]->setBinContent( ic, mean06 );
              meaopn03_[ism-1]->setBinError( ic, rms06 );
            } else {
              meaopn03_[ism-1]->setEntries( 1.+meaopn03_[ism-1]->getEntries() );
            }
          }

        }

        if ( update08 ) {

          int ic = Numbers::icEB(ism, ie, ip);

          if ( meaopn04_[ism-1] ) {
            if ( mean08 > 0. ) {
              meaopn04_[ism-1]->setBinContent( ic, mean08 );
              meaopn04_[ism-1]->setBinError( ic, rms08 );
            } else {
              meaopn04_[ism-1]->setEntries( 1.+meaopn04_[ism-1]->getEntries() );
            }
          }

        }

        if ( update09 ) {

          int ic = Numbers::icEB(ism, ie, ip);

          if ( met01_[ism-1] ) {
            if ( mean09 > 0. ) {
              met01_[ism-1]->setBinContent( ic, mean09 );
              met01_[ism-1]->setBinError( ic, rms09 );
            } else {
              met01_[ism-1]->setEntries(1.+met01_[ism-1]->getEntries());
            }
          }

          if ( metav01_[ism-1] ) metav01_[ism-1] ->Fill(mean09);
          if ( metrms01_[ism-1] ) metrms01_[ism-1]->Fill(rms09);

        }

        if ( update10 ) {

          int ic = Numbers::icEB(ism, ie, ip);

          if ( met02_[ism-1] ) {
            if ( mean10 > 0. ) {
              met02_[ism-1]->setBinContent( ic, mean10 );
              met02_[ism-1]->setBinError( ic, rms10 );
            } else {
              met02_[ism-1]->setEntries(1.+met02_[ism-1]->getEntries());
            }
          }

          if ( metav02_[ism-1] ) metav02_[ism-1] ->Fill(mean10);
          if ( metrms02_[ism-1] ) metrms02_[ism-1]->Fill(rms10);

        }

        if ( update11 ) {

          int ic = Numbers::icEB(ism, ie, ip);

          if ( met03_[ism-1] ) {
            if ( mean11 > 0. ) {
              met03_[ism-1]->setBinContent( ic, mean11 );
              met03_[ism-1]->setBinError( ic, rms11 );
            } else {
              met03_[ism-1]->setEntries(1.+met03_[ism-1]->getEntries());
            }
          }

          if ( metav03_[ism-1] ) metav03_[ism-1] ->Fill(mean11);
          if ( metrms03_[ism-1] ) metrms03_[ism-1]->Fill(rms11);

        }

        if ( update12 ) {

          int ic = Numbers::icEB(ism, ie, ip);

          if ( met04_[ism-1] ) {
            if ( mean12 > 0. ) {
              met04_[ism-1]->setBinContent( ic, mean12 );
              met04_[ism-1]->setBinError( ic, rms12 );
            } else {
              met04_[ism-1]->setEntries(1.+met04_[ism-1]->getEntries());
            }
          }

          if ( metav04_[ism-1] ) metav04_[ism-1] ->Fill(mean12);
          if ( metrms04_[ism-1] ) metrms04_[ism-1]->Fill(rms12);

        }

        // masking

        if ( mask1.size() != 0 ) {
          map<EcalLogicID, RunCrystalErrorsDat>::const_iterator m;
          for (m = mask1.begin(); m != mask1.end(); m++) {

            EcalLogicID ecid = m->first;

            int ic = Numbers::indexEB(ism, ie, ip);

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EB_crystal_number", Numbers::iSM(ism, EcalBarrel), ic).getLogicID() ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                UtilsClient::maskBinContent( meg01_[ism-1], ie, ip );
                UtilsClient::maskBinContent( meg02_[ism-1], ie, ip );
                UtilsClient::maskBinContent( meg03_[ism-1], ie, ip );
                UtilsClient::maskBinContent( meg04_[ism-1], ie, ip );
              }
            }

          }
        }

        // TT masking

        if ( mask3.size() != 0 ) {
          map<EcalLogicID, RunTTErrorsDat>::const_iterator m;
          for (m = mask3.begin(); m != mask3.end(); m++) {

            EcalLogicID ecid = m->first;

            int itt = Numbers::iSC(ism, EcalBarrel, ie, ip);

            if ( ecid.getLogicID() == LogicID::getEcalLogicID("EB_trigger_tower", Numbers::iSM(ism, EcalBarrel), itt).getLogicID() ) {
              if ( (m->second).getErrorBits() & bits01 ) {
                UtilsClient::maskBinContent( meg01_[ism-1], ie, ip );
                UtilsClient::maskBinContent( meg02_[ism-1], ie, ip );
                UtilsClient::maskBinContent( meg03_[ism-1], ie, ip );
                UtilsClient::maskBinContent( meg04_[ism-1], ie, ip );
              }
            }

          }
        }

      }
    }

    for ( int i = 1; i <= 10; i++ ) {

      if ( meg05_[ism-1] ) meg05_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg06_[ism-1] ) meg06_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg07_[ism-1] ) meg07_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg08_[ism-1] ) meg08_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg09_[ism-1] ) meg09_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg10_[ism-1] ) meg10_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg11_[ism-1] ) meg11_[ism-1]->setBinContent( i, 1, 2. );
      if ( meg12_[ism-1] ) meg12_[ism-1]->setBinContent( i, 1, 2. );

      bool update01;
      bool update02;
      bool update03;
      bool update04;
      bool update05;
      bool update06;
      bool update07;
      bool update08;
      bool update09;
      bool update10;
      bool update11;
      bool update12;
      bool update13;
      bool update14;
      bool update15;
      bool update16;

      float num01, num02, num03, num04, num05, num06, num07, num08;
      float num09, num10, num11, num12, num13, num14, num15, num16;
      float mean01, mean02, mean03, mean04, mean05, mean06, mean07, mean08;
      float mean09, mean10, mean11, mean12, mean13, mean14, mean15, mean16;
      float rms01, rms02, rms03, rms04, rms05, rms06, rms07, rms08;
      float rms09, rms10, rms11, rms12, rms13, rms14, rms15, rms16;

      update01 = UtilsClient::getBinStatistics(i01_[ism-1], i, 0, num01, mean01, rms01);
      update02 = UtilsClient::getBinStatistics(i02_[ism-1], i, 0, num02, mean02, rms02);
      update03 = UtilsClient::getBinStatistics(i03_[ism-1], i, 0, num03, mean03, rms03);
      update04 = UtilsClient::getBinStatistics(i04_[ism-1], i, 0, num04, mean04, rms04);
      update05 = UtilsClient::getBinStatistics(i05_[ism-1], i, 0, num05, mean05, rms05);
      update06 = UtilsClient::getBinStatistics(i06_[ism-1], i, 0, num06, mean06, rms06);
      update07 = UtilsClient::getBinStatistics(i07_[ism-1], i, 0, num07, mean07, rms07);
      update08 = UtilsClient::getBinStatistics(i08_[ism-1], i, 0, num08, mean08, rms08);
      update09 = UtilsClient::getBinStatistics(i09_[ism-1], i, 0, num09, mean09, rms09);
      update10 = UtilsClient::getBinStatistics(i10_[ism-1], i, 0, num10, mean10, rms10);
      update11 = UtilsClient::getBinStatistics(i11_[ism-1], i, 0, num11, mean11, rms11);
      update12 = UtilsClient::getBinStatistics(i12_[ism-1], i, 0, num12, mean12, rms12);
      update13 = UtilsClient::getBinStatistics(i13_[ism-1], i, 0, num13, mean13, rms13);
      update14 = UtilsClient::getBinStatistics(i14_[ism-1], i, 0, num14, mean14, rms14);
      update15 = UtilsClient::getBinStatistics(i15_[ism-1], i, 0, num15, mean15, rms15);
      update16 = UtilsClient::getBinStatistics(i16_[ism-1], i, 0, num16, mean16, rms16);

      if ( update01 && update05 ) {

        float val;

        val = 1.;
        if ( mean01 < amplitudeThresholdPnG01_ )
          val = 0.;
        if ( mean05 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
             pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean05)
          val = 0.;
        if ( rms05 > pedPnRMSThreshold_[0] )
          val = 0.;

        if ( meg05_[ism-1] )           meg05_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms01_[ism-1] ) mepnprms01_[ism-1]->Fill(rms05);

      }

      if ( update02 && update06 ) {

       float val;

        val = 1.;
        if ( mean02 < amplitudeThresholdPnG01_ )
          val = 0.;
        if ( mean06 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
             pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean06)
          val = 0.;
        if ( rms06 > pedPnRMSThreshold_[0] )
          val = 0.;

        if ( meg06_[ism-1] )           meg06_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms02_[ism-1] ) mepnprms02_[ism-1]->Fill(rms06);
      }

      if ( update03 && update07 ) {

        float val;

        val = 1.;
        if ( mean03 < amplitudeThresholdPnG01_ )
          val = 0.;
        if ( mean07 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
             pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean07)
          val = 0.;
        if ( rms07 > pedPnRMSThreshold_[0] )
          val = 0.;

        if ( meg07_[ism-1] )           meg07_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms03_[ism-1] ) mepnprms03_[ism-1]->Fill(rms07);
      }

      if ( update04 && update08 ) {

        float val;

        val = 1.;
        if ( mean04 < amplitudeThresholdPnG01_ )
          val = 0.;
        if ( mean08 <  pedPnExpectedMean_[0] - pedPnDiscrepancyMean_[0] ||
             pedPnExpectedMean_[0] + pedPnDiscrepancyMean_[0] < mean08)
          val = 0.;
        if ( rms08 > pedPnRMSThreshold_[0] )
          val = 0.;

        if ( meg08_[ism-1] )           meg08_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms04_[ism-1] ) mepnprms04_[ism-1]->Fill(rms08);
      }

      if ( update09 && update13 ) {

        float val;

        val = 1.;
        if ( mean09 < amplitudeThresholdPnG16_ )
          val = 0.;
        if ( mean13 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
             pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean13)
          val = 0.;
        if ( rms13 > pedPnRMSThreshold_[1] )
          val = 0.;

        if ( meg09_[ism-1] )           meg09_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms05_[ism-1] ) mepnprms05_[ism-1]->Fill(rms13);
      }

      if ( update10 && update14 ) {

        float val;

        val = 1.;
        if ( mean10 < amplitudeThresholdPnG16_ )
          val = 0.;
        if ( mean14 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
             pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean14)
          val = 0.;
        if ( rms14 > pedPnRMSThreshold_[1] )
          val = 0.;

        if ( meg10_[ism-1] )           meg10_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms06_[ism-1] ) mepnprms06_[ism-1]->Fill(rms14);
      }

      if ( update11 && update15 ) {

        float val;

        val = 1.;
        if ( mean11 < amplitudeThresholdPnG16_ )
          val = 0.;
        if ( mean15 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
             pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean15)
          val = 0.;
        if ( rms15 > pedPnRMSThreshold_[1] )
          val = 0.;

        if ( meg11_[ism-1] )           meg11_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms07_[ism-1] ) mepnprms07_[ism-1]->Fill(rms15);
      }

      if ( update12 && update16 ) {

        float val;

        val = 1.;
        if ( mean12 < amplitudeThresholdPnG16_ )
          val = 0.;
        if ( mean16 <  pedPnExpectedMean_[1] - pedPnDiscrepancyMean_[1] ||
             pedPnExpectedMean_[1] + pedPnDiscrepancyMean_[1] < mean16)
          val = 0.;
        if ( rms16 > pedPnRMSThreshold_[1] )
          val = 0.;

        if ( meg12_[ism-1] )           meg12_[ism-1]->setBinContent(i, 1, val);
        if ( mepnprms08_[ism-1] ) mepnprms08_[ism-1]->Fill(rms16);
      }

      // masking

      if ( mask2.size() != 0 ) {
        map<EcalLogicID, RunPNErrorsDat>::const_iterator m;
        for (m = mask2.begin(); m != mask2.end(); m++) {

          EcalLogicID ecid = m->first;

          if ( ecid.getLogicID() == LogicID::getEcalLogicID("EB_LM_PN", Numbers::iSM(ism, EcalBarrel), i-1).getLogicID() ) {
            if ( (m->second).getErrorBits() & (bits01|bits02) ) {
              UtilsClient::maskBinContent( meg05_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg06_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg07_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg08_[ism-1], i, 1 );
            }
            if ( (m->second).getErrorBits() & (bits01|bits04) ) {
              UtilsClient::maskBinContent( meg09_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg10_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg11_[ism-1], i, 1 );
              UtilsClient::maskBinContent( meg12_[ism-1], i, 1 );
            }
          }

        }
      }

    }

    for ( int i = 1; i <= 10; i++ ) {

      if ( hs01_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs01_[ism-1] );
        if ( me_hs01_[ism-1] ) {
          me_hs01_[ism-1]->setBinContent( i, hs01_[ism-1]->GetBinContent(ic, i) );
          me_hs01_[ism-1]->setBinError( i, hs01_[ism-1]->GetBinError(ic, i) );
        }
      }

      if ( hs02_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs02_[ism-1] );
        if ( me_hs02_[ism-1] ) {
          me_hs02_[ism-1]->setBinContent( i, hs02_[ism-1]->GetBinContent(ic, i) );
          me_hs02_[ism-1]->setBinError( i, hs02_[ism-1]->GetBinError(ic, i) );
        }
      }

      if ( hs03_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs03_[ism-1] );
        if ( me_hs03_[ism-1] ) {
          me_hs03_[ism-1]->setBinContent( i, hs03_[ism-1]->GetBinContent(ic, i) );
          me_hs03_[ism-1]->setBinError( i, hs03_[ism-1]->GetBinError(ic, i) );
        }
      }

      if ( hs04_[ism-1] ) {
        int ic = UtilsClient::getFirstNonEmptyChannel( hs04_[ism-1] );
        if ( me_hs04_[ism-1] ) {
          me_hs04_[ism-1]->setBinContent( i, hs04_[ism-1]->GetBinContent(ic, i) );
          me_hs04_[ism-1]->setBinError( i, hs04_[ism-1]->GetBinError(ic, i) );
        }
      }

    }

  }

}

void EBLaserClient::softReset(bool flag) {

}

