import FWCore.ParameterSet.Config as cms
from RecoMET.METPUSubtraction.deepMetSonicProducer_cff import sonic_deepmet

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
process = cms.Process('DeepMET',enableSonicTriton)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames=cms.untracked.vstring(
                                'file:/storage/local/data1/home/yfeng/DeepMET/CMSSW_11_2_0_pre10_Patatrack/src/RecoMET/METPUSubtraction/test/9E5D0032-E9FB-6646-B1AB-67EA8B95FCCD.root'
                            ),
                            )

process.load("HeterogeneousCore.SonicTriton.TritonService_cff")
process.TritonService.verbose = True
# fallback server
process.TritonService.fallback.verbose = False
process.TritonService.fallback.useDocker = False
process.TritonService.fallback.useGPU = True
process.TritonService.servers.append(
    cms.PSet(
        name = cms.untracked.string("default"),
        address = cms.untracked.string("0.0.0.0"),
        port = cms.untracked.uint32(8001),
    )
)


process.deepMETProducer = sonic_deepmet.clone(
)

process.p = cms.Path()
process.p += process.deepMETProducer

process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands=cms.untracked.vstring(
                                      'keep *'),
                                  fileName=cms.untracked.string(
                                      "DeepMETTest.root")
                                  )
process.outpath  = cms.EndPath(process.output)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True ),
    numberOfThreads = cms.untracked.uint32( 4 ),
    numberOfStreams = cms.untracked.uint32( 0 ),
    sizeOfStackForThreadsInKB = cms.untracked.uint32( 10*1024 )
)
