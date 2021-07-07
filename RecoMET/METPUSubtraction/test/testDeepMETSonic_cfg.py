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
    input=cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
                            fileNames=cms.untracked.vstring(
                                #'/store/mc/RunIISummer20UL18MiniAOD/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v11_L1v1-v1/260000/03794341-C401-CC45-B5FC-D11264E449CE.root'
                                'file:/storage/local/data1/home/yfeng/SONIC/CMSSW_12_0_0_pre1/src/RecoTauTag/RecoTau/test/A256C80D-0943-E811-998E-7CD30AB0522C.root'
                            ),
                            )

process.load("HeterogeneousCore.SonicTriton.TritonService_cff")
process.TritonService.verbose = True
process.TritonService.fallback.useDocker = True
process.TritonService.fallback.verbose = True
# uncomment this part if there is one server running at 0.0.0.0 with grpc port 8001
process.TritonService.servers.append(
    cms.PSet(
        name = cms.untracked.string("default"),
        address = cms.untracked.string("0.0.0.0"),
        port = cms.untracked.uint32(8021),
    )
)


process.deepMETProducer = sonic_deepmet.clone(
)
process.deepMETProducer.Client.verbose = cms.untracked.bool(True)

process.p = cms.Path()
process.p += process.deepMETProducer

process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands=cms.untracked.vstring(
                                      'keep *'),
                                  fileName=cms.untracked.string(
                                      "DeepMETTestSonic.root"
                                      )
                                  )
process.outpath  = cms.EndPath(process.output)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True ),
    numberOfThreads = cms.untracked.uint32( 1 ),
    numberOfStreams = cms.untracked.uint32( 0 ),
    sizeOfStackForThreadsInKB = cms.untracked.uint32( 10*1024 )
)

process.FastTimerService = cms.Service( "FastTimerService",
    dqmPath = cms.untracked.string( "HLT/TimerService" ),
    dqmModuleTimeRange = cms.untracked.double( 40.0 ),
    enableDQMbyPath = cms.untracked.bool( True ),
    writeJSONSummary = cms.untracked.bool( True ),
    dqmPathMemoryResolution = cms.untracked.double( 5000.0 ),
    enableDQM = cms.untracked.bool( True ),
    enableDQMbyModule = cms.untracked.bool( True ),
    dqmModuleMemoryRange = cms.untracked.double( 100000.0 ),
    dqmModuleMemoryResolution = cms.untracked.double( 500.0 ),
    dqmMemoryResolution = cms.untracked.double( 5000.0 ),
    enableDQMbyLumiSection = cms.untracked.bool( True ),
    dqmPathTimeResolution = cms.untracked.double( 0.5 ),
    printEventSummary = cms.untracked.bool( False ),
    dqmPathTimeRange = cms.untracked.double( 100.0 ),
    dqmTimeRange = cms.untracked.double( 2000.0 ),
    enableDQMTransitions = cms.untracked.bool( False ),
    dqmPathMemoryRange = cms.untracked.double( 1000000.0 ),
    dqmLumiSectionsRange = cms.untracked.uint32( 2500 ),
    enableDQMbyProcesses = cms.untracked.bool( True ),
    dqmMemoryRange = cms.untracked.double( 1000000.0 ),
    dqmTimeResolution = cms.untracked.double( 5.0 ),
    printRunSummary = cms.untracked.bool( False ),
    dqmModuleTimeResolution = cms.untracked.double( 0.2 ),
    printJobSummary = cms.untracked.bool( True ),
    jsonFileName = cms.untracked.string( "result_sonic.json" )
)

process.ThroughputService = cms.Service( "ThroughputService",
    dqmPath = cms.untracked.string( "HLT/Throughput" ),
    eventRange = cms.untracked.uint32( 10000 ),
    timeRange = cms.untracked.double( 60000.0 ),
    printEventSummary = cms.untracked.bool( True ),
    eventResolution = cms.untracked.uint32( 100 ),
    enableDQM = cms.untracked.bool( True ),
    dqmPathByProcesses = cms.untracked.bool( True ),
    timeResolution = cms.untracked.double( 5.828 )
)

##
process.load('FWCore.MessageLogger.MessageLogger_cfi')
if process.maxEvents.input.value()>10:
     process.MessageLogger.cerr.FwkReport.reportEvery = process.maxEvents.input.value()//10
if process.maxEvents.input.value()>2000 or process.maxEvents.input.value()<0:
     process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.cerr.ThroughputService = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000),
    reportEvery = cms.untracked.int32(1)
)
for msg in ['TritonClient', 'TritonService', 'deepMETProducer', 'deepMETProducer:TritonClient']:
    setattr(process.MessageLogger.cerr,msg,
        cms.untracked.PSet(
            limit = cms.untracked.int32(10000000),
        )
    )
