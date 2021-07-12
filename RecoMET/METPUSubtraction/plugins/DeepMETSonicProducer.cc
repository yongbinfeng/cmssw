#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"

class DeepMETSonicProducer : public TritonEDProducer<> {
public:
  explicit DeepMETSonicProducer(const edm::ParameterSet& cfg)
      : TritonEDProducer<>(cfg, "DeepMETSonicProducer"),
        pfName_(cfg.getParameter<edm::InputTag>("pf_src")),
        pf_token_(consumes<std::vector<pat::PackedCandidate>>(pfName_)),
        norm_(50.0),
        ignore_leptons_(false),
        max_n_pf_(4500),
        px_leptons_(0),
        py_leptons_(0) {
    produces<pat::METCollection>();
  }
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  float scale_and_rm_outlier(float val, float scale);

private:
  edm::InputTag pfName_;
  const edm::EDGetTokenT<std::vector<pat::PackedCandidate>> pf_token_;
  const float norm_;
  const bool ignore_leptons_;
  const unsigned int max_n_pf_;
  float px_leptons_;
  float py_leptons_;

  inline static const std::unordered_map<int, int32_t> charge_embedding_{{-1, 0}, {0, 1}, {1, 2}};
  inline static const std::unordered_map<int, int32_t> pdg_id_embedding_{
      {-211, 0}, {-13, 1}, {-11, 2}, {0, 3}, {1, 4}, {2, 5}, {11, 6}, {13, 7}, {22, 8}, {130, 9}, {211, 10}};
};

float DeepMETSonicProducer::scale_and_rm_outlier(float val, float scale) {
  float ret_val = val * scale;
  if (ret_val > 1e6 || ret_val < -1e6)
    return 0.;
  return ret_val;
}

void DeepMETSonicProducer::acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) {
  // one event per batch
  client_->setBatchSize(1);
  px_leptons_ = 0.;
  py_leptons_ = 0.;
  const float scale = 1. / norm_;

  auto const& pfs = iEvent.get(pf_token_);

  auto& input = iInput.at("input");
  auto pfdata = input.allocate<float>();
  auto& vpfdata = (*pfdata)[0];

  auto& input_cat0 = iInput.at("input_cat0");
  auto pfchg = input_cat0.allocate<float>();
  auto& vpfchg = (*pfchg)[0];

  auto& input_cat1 = iInput.at("input_cat1");
  auto pfpdgId = input_cat1.allocate<float>();
  auto& vpfpdgId = (*pfpdgId)[0];

  auto& input_cat2 = iInput.at("input_cat2");
  auto pffromPV = input_cat2.allocate<float>();
  auto& vpffromPV = (*pffromPV)[0];

  size_t i_pf = 0;
  for (const auto& pf : pfs) {
    if (ignore_leptons_) {
      int pdg_id = std::abs(pf.pdgId());
      if (pdg_id == 11 || pdg_id == 13) {
        px_leptons_ += pf.px();
        py_leptons_ += pf.py();
        continue;
      }
    }

    // PF keys [b'PF_dxy', b'PF_dz', b'PF_eta', b'PF_mass', b'PF_pt', b'PF_puppiWeight', b'PF_px', b'PF_py']
    vpfdata.push_back(pf.dxy());
    vpfdata.push_back(pf.dz());
    vpfdata.push_back(pf.eta());
    vpfdata.push_back(pf.mass());
    vpfdata.push_back(scale_and_rm_outlier(pf.pt(), scale));
    vpfdata.push_back(pf.puppiWeight());
    vpfdata.push_back(scale_and_rm_outlier(pf.px(), scale));
    vpfdata.push_back(scale_and_rm_outlier(pf.py(), scale));

    vpfchg.push_back(charge_embedding_.at(pf.charge()));

    vpfpdgId.push_back(pdg_id_embedding_.at(pf.pdgId()));

    vpffromPV.push_back(pf.fromPV());

    ++i_pf;
    if (i_pf == max_n_pf_) {
      break;  // output a warning?
    }
  }

  // fill the remaining with zeros
  while (i_pf < 4500) {
    for (int i = 0; i < 8; i++) {
      vpfdata.push_back(0.);
    }
    vpfchg.push_back(0);
    vpfpdgId.push_back(0);
    vpffromPV.push_back(0);
    i_pf++;
  }

  input.toServer(pfdata);
  input_cat0.toServer(pfchg);
  input_cat1.toServer(pfpdgId);
  input_cat2.toServer(pffromPV);
}

void DeepMETSonicProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) {
  const auto& output1 = iOutput.begin()->second;
  const auto& outputs = output1.fromServer<float>();

  // outputs are px and py
  float px = outputs[0][0] * norm_;
  float py = outputs[0][1] * norm_;

  // subtract the lepton pt contribution
  px -= px_leptons_;
  py -= py_leptons_;

  auto pf_mets = std::make_unique<pat::METCollection>();
  const reco::Candidate::LorentzVector p4(px, py, 0., std::hypot(px, py));
  pf_mets->emplace_back(reco::MET(p4, {}));
  iEvent.put(std::move(pf_mets));
}

void DeepMETSonicProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  TritonClient::fillPSetDescription(desc);
  desc.add<unsigned>("batchSize", 1);
  desc.add<edm::InputTag>("pf_src");
  descriptions.add("deepMETSonicProducer", desc);
}

DEFINE_FWK_MODULE(DeepMETSonicProducer);
