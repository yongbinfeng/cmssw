#ifndef SHALLOW_CLUSTERS_PRODUCER
#define SHALLOW_CLUSTERS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/DetSetVector.h"
class SiStripClusterInfo;
class SiStripProcessedRawDigi;

class ShallowClustersProducer : public edm::EDProducer {
  
 public:

  explicit ShallowClustersProducer(const edm::ParameterSet&);

 private:

  edm::InputTag theClustersLabel;
  std::string Prefix;
  void produce( edm::Event &, const edm::EventSetup & );

  struct moduleVars {
    moduleVars(uint32_t);
    int subdetid, side, layerwheel, stringringrod, petal, stereo;
    uint32_t module;
  };

  struct NearDigis { 
    NearDigis(const SiStripClusterInfo&);
    NearDigis(const SiStripClusterInfo&, const edm::DetSetVector<SiStripProcessedRawDigi>&);
    float max, left, right, first, last, Lleft, Rright; 
    float etaX() const {return (left+right)/max/2.;}
    float eta()  const {return right>left ? max/(max+right) : left/(left+max);}
    float etaasymm() const {return right>left ? (right-max)/(right+max) : (max-left)/(max+left);}
    float outsideasymm() const {return (last-first)/(last+first);}
  };

};

#endif
