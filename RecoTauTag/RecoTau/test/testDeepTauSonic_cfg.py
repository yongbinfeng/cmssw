"""
cleaner version to run the DeepTau Sonic Producer only
for debugging performance
"""

## working points for the discriminator

workingPoints_ = {
    "e": {
        "VVVLoose": 0.0630386,
        "VVLoose": 0.1686942,
        "VLoose": 0.3628130,
        "Loose": 0.6815435,
        "Medium": 0.8847544,
        "Tight": 0.9675541,
        "VTight": 0.9859251,
        "VVTight": 0.9928449,
    },
    "mu": {
        "VLoose": 0.1058354,
        "Loose": 0.2158633,
        "Medium": 0.5551894,
        "Tight": 0.8754835,
    },
    "jet": {
        "VVVLoose": 0.2599605,
        "VVLoose": 0.4249705,
        "VLoose": 0.5983682,
        "Loose": 0.7848675,
        "Medium": 0.8834768,
        "Tight": 0.9308689,
        "VTight": 0.9573137,
        "VVTight": 0.9733927,
    },
}

def processDeepProducer(process, producer_name):
    postfix = ""
    import six
    for target,points in six.iteritems(workingPoints_):
        cut_expressions = []
        for index, (point,cut) in enumerate(six.iteritems(points)):
            cut_expressions.append(str(cut))

        setattr(getattr(process, producer_name+postfix), 'VS{}WP'.format(target), cms.vstring(*cut_expressions))


import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton
from RecoTauTag.RecoTau.deeptauIdSonicProducer_cff import sonic_deeptau

process = cms.Process('DeepTau', enableSonicTriton)

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
                                #'file:/storage/local/data1/home/yfeng/DeepMET/CMSSW_11_2_0_pre10_Patatrack/src/RecoMET/METPUSubtraction/test/9E5D0032-E9FB-6646-B1AB-67EA8B95FCCD.root'
                                'file:/storage/local/data1/home/yfeng/SONIC/CMSSW_12_0_0_pre1/src/RecoTauTag/RecoTau/test/A256C80D-0943-E811-998E-7CD30AB0522C.root'
                            ),
                            )

process.load("HeterogeneousCore.SonicTriton.TritonService_cff")
process.TritonService.verbose = True
process.TritonService.servers.append(
    cms.PSet(
        name = cms.untracked.string("default"),
        address = cms.untracked.string("0.0.0.0"),
        port = cms.untracked.uint32(8021),
    )
)


process.deepTauProducer = sonic_deeptau.clone(
)
# add working points to the process
processDeepProducer(process, 'deepTauProducer')

process.p = cms.Path()
process.p += process.deepTauProducer

process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands=cms.untracked.vstring(
                                      'keep *'),
                                  fileName=cms.untracked.string(
                                      "DeepTauSonicTest.root")
                                  )
process.outpath  = cms.EndPath(process.output)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( True ),
    numberOfThreads = cms.untracked.uint32( 1 ),
    numberOfStreams = cms.untracked.uint32( 0 ),
    #sizeOfStackForThreadsInKB = cms.untracked.uint32( 10*1024 )
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
process.MessageLogger.cerr.TritonClient = cms.untracked.PSet(
    limit = cms.untracked.int32(100000000),
)
process.MessageLogger.cerr.TritonService = cms.untracked.PSet(
    limit = cms.untracked.int32(100000000),
)
process.MessageLogger.cerr.deepTauProducer = cms.untracked.PSet(
    limit = cms.untracked.int32(100000000),
)
setattr(process.MessageLogger.cerr, "deepTauProducer:TritonClient",
    cms.untracked.PSet(
    limit = cms.untracked.int32(100000000),
    )
)


