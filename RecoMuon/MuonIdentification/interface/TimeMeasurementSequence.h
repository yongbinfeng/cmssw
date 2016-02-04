#ifndef MuonIdentification_TimeMeasurementSequence_h
#define MuonIdentification_TimeMeasurementSequence_h

/** \class reco::TimeMeasurementSequence TimeMeasurementSequence.h RecoMuon/MuonIdentification/interface/TimeMeasurementSequence.h
 *  
 * A class holding a set of individual time measurements along the muon trajectory
 *
 * \author Piotr Traczyk, CERN
 *
 * \version $Id: TimeMeasurementSequence.h,v 1.5 2011/02/24 15:41:53 farrell3 Exp $
 *
 */

class TimeMeasurementSequence {

    public:

      std::vector <double> dstnc;
      std::vector <double> local_t0;
      std::vector <double> weightVertex;
      std::vector <double> weightInvbeta;
      
      double totalWeightInvbeta;
      double totalWeightVertex;
      
      TimeMeasurementSequence():
	totalWeightInvbeta(0),
	totalWeightVertex(0)
	 {}

};


#endif
