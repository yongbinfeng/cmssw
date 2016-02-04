import FWCore.ParameterSet.Config as cms

#                                __                 __ 
#    ____  ____   ____   _______/  |______    _____/  |_  ______
#  _/ ___\/  _ \ /    \ /  ___/\   __\__  \  /    \   __\/  ___/
#  \  \__(  <_> )   |  \\___ \  |  |  / __ \|   |  \  |  \___ \
#   \___  >____/|___|  /____  > |__| (____  /___|  /__| /____  >
#        \/           \/     \/            \/     \/          \/
        
PHOTON_CALOIDISO_ET_HIGH_CUT_MIN = 26.
PHOTON_CALOIDISO_ET_LOW_CUT_MIN = 18.
MASS_DIPHOTON_CALOIDISO_CUT_MIN = 60.

PHOTON_R9ID_ET_HIGH_CUT_MIN = 22.
PHOTON_R9ID_ET_LOW_CUT_MIN = 18.
MASS_DIPHOTON_R9ID_CUT_MIN = 60.

#  _____ __  _____            _         _   _
# |  |  |  ||_   _|   ___ ___| |___ ___| |_|_|___ ___
# |     |  |__| |    |_ -| -_| | -_|  _|  _| | . |   |
# |__|__|_____|_|    |___|___|_|___|___|_| |_|___|_|_|
                                                     
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
DiPhotonHltFilter = copy.deepcopy(hltHighLevel)
DiPhotonHltFilter.throw = cms.bool(False)
DiPhotonHltFilter.HLTPaths = ["HLT_Photon*_Photon*"]

#  _____     _     _____   _ _____
# |     |___| |___|     |_| |     |___ ___
# |   --| .'| | . |-   -| . |-   -|_ -| . |
# |_____|__,|_|___|_____|___|_____|___|___|
                                          
hltDiPhotonCaloIdIsoObjectProducer = cms.EDProducer("CandidateTriggerObjectProducer",
                                             triggerName = cms.string("HLT_Photon.*_CaloId.*_Iso.*_Photon.*_CaloId.*_Iso.*_.*"),
                                             triggerResults = cms.InputTag("TriggerResults","","HLT"),
                                             triggerEvent   = cms.InputTag("hltTriggerSummaryAOD","","HLT")
                                             )


TrailingPtCaloIdIsoPhotons = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("hltDiPhotonCaloIdIsoObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_CALOIDISO_ET_LOW_CUT_MIN))
)

LeadingPtCaloIdIsoPhotons = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("hltDiPhotonCaloIdIsoObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_CALOIDISO_ET_HIGH_CUT_MIN))
)

CaloIdIsoPhotonPairs = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("LeadingPtCaloIdIsoPhotons TrailingPtCaloIdIsoPhotons"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string("mass > " + str(MASS_DIPHOTON_CALOIDISO_CUT_MIN))
)

CaloIdIsoPhotonPairsCounter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("CaloIdIsoPhotonPairs"),
                                    minNumber = cms.uint32(1)
                                    )

CaloIdIsoPhotonPairsFilter = cms.Sequence(DiPhotonHltFilter*hltDiPhotonCaloIdIsoObjectProducer*TrailingPtCaloIdIsoPhotons*LeadingPtCaloIdIsoPhotons*CaloIdIsoPhotonPairs*CaloIdIsoPhotonPairsCounter)

#  _____ ___ _____   _
# | __  | . |     |_| |
# |    -|_  |-   -| . |
# |__|__|___|_____|___|
 
hltDiPhotonR9IdObjectProducer = hltDiPhotonCaloIdIsoObjectProducer.clone(
                                             triggerName = cms.string("HLT_Photon.*_R9Id.*_Photon.*_R9Id.*_.*"),
                                             )

TrailingPtR9IdPhotons = TrailingPtCaloIdIsoPhotons.clone(
    src = cms.InputTag("hltDiPhotonR9IdObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_R9ID_ET_LOW_CUT_MIN))
)

LeadingPtR9IdPhotons = LeadingPtCaloIdIsoPhotons.clone(
    src = cms.InputTag("hltDiPhotonR9IdObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_R9ID_ET_LOW_CUT_MIN))
)

R9IdPhotonPairs = CaloIdIsoPhotonPairs.clone( 
    decay = cms.string("LeadingPtR9IdPhotons TrailingPtR9IdPhotons"), # charge coniugate states are implied
    cut   = cms.string("mass > " + str(MASS_DIPHOTON_R9ID_CUT_MIN))
)

R9IdPhotonPairsCounter = CaloIdIsoPhotonPairsCounter.clone(
                                    src = cms.InputTag("R9IdPhotonPairs"),
                                    )

R9IdPhotonPairsFilter = cms.Sequence(DiPhotonHltFilter*hltDiPhotonR9IdObjectProducer*TrailingPtR9IdPhotons*LeadingPtR9IdPhotons*R9IdPhotonPairs*R9IdPhotonPairsCounter)
