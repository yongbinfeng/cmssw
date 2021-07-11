import FWCore.ParameterSet.Config as cms
from RecoMET.METPUSubtraction.deepMETProducer_cfi import deepMETProducer

process = cms.Process('DeepMET')

process.task = cms.Task()

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
                            secondaryFileNames=cms.untracked.vstring(),
                            fileNames=cms.untracked.vstring(
                                '/store/relval/CMSSW_11_0_0_pre7/RelValTTbar_13/MINIAODSIM/PUpmx25ns_110X_mc2017_realistic_v1_rsb-v1/10000/A745F01B-D6C3-A843-97B7-2B12C7C0DD4E.root'
                            ),
                            skipEvents=cms.untracked.uint32(0)
                            )

process.deepMETProducer = deepMETProducer.clone()

process.sequence = cms.Sequence(process.deepMETProducer)
process.p = cms.Path(process.sequence)
process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands=cms.untracked.vstring(
                                      'keep *'),
                                  fileName=cms.untracked.string(
                                      "DeepMETTest.root")
                                  )
process.outpath  = cms.EndPath(process.output)
