# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: NANO -s NANO --nThreads 2 --data --era Run2_2017,run2_nanoAOD_106Xv1 --conditions 106X_dataRun2_v32 --eventcontent NANOAOD --datatier NANOAOD --filein file:pippo.root -n 100 --python_filename=nano_Run2017.py --no_exec --customise_commands=process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)))
import FWCore.ParameterSet.Config as cms

import FWCore.ParameterSet.VarParsing as VarParsing
import re

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.Modifier_run2_nanoAOD_LowPU_cff import run2_nanoAOD_LowPU

#process = cms.Process('NANO',Run2_2017, run2_nanoAOD_LowPU)
process = cms.Process('NANO',run2_nanoAOD_LowPU)

opt = VarParsing.VarParsing('analysis')
opt.register('isMC', -1, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, 'Flag indicating if the input samples are from MC (1) or from the detector (0).')
opt.register('filterTrig', 0, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.bool, 'Flag indiciating to apply the trigger filter or not. Default is 0.')
opt.parseArguments()


# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('PhysicsTools.NanoAOD.nano_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(200)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # test data
        'file:/afs/cern.ch/work/y/yofeng/public/WpT/CMSSW_10_6_20/src/00405246-A939-E811-A3F3-801844DEEC30.root'
        #'file:/afs/cern.ch/work/y/yofeng/public/WpT/CMSSW_10_6_20/src/SingleMuon_H.root'
        # test MC
        #'file:/afs/cern.ch/work/y/yofeng/public/WpT/CMSSW_10_6_20/src/02036C45-98AC-E911-8DEC-1866DAEA79D0.root'
    ),
    secondaryFileNames = cms.untracked.vstring()
)

if opt.isMC < 0 and len(process.source.fileNames) > 0:
  if re.match(r'.*/(MINI)?AODSIM/.*', process.source.fileNames[0]):
    print "MC dataset detected."
    opt.isMC = 1
  elif re.match(r'.*/(MINI)?AOD/.*', process.source.fileNames[0]):
    print "Real data dataset detected."
    opt.isMC = 0

if opt.isMC < 0:
  raise Exception("Failed to detect data type. Data type need to be specify with the isMC cmsRun command line option")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('NANO nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition
if opt.isMC:
    datatier = 'NANOAODSIM'
    eventcontent = process.NANOAODSIMEventContent
    outputname = "NanoAOD_MC.root"
    globaltag = "94X_mc2017_realistic_v14"
    seq = process.nanoSequenceMC
else:
    datatier = 'NANOAOD'
    eventcontent = process.NANOAODEventContent
    outputname = "NanoAOD_Data.root"
    globaltag = "94X_dataRun2_ReReco_EOY17_v6"
    seq = process.nanoSequence

process.NANOAODoutput = cms.OutputModule("NanoAODOutputModule",
    compressionAlgorithm = cms.untracked.string('LZMA'),
    compressionLevel = cms.untracked.int32(9),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string(datatier),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(outputname),
    outputCommands = eventcontent.outputCommands,
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('nanoAOD_step')
    )
)

# Additional output definition

# Other statements
print("global tag: ", globaltag)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, globaltag, '')

# add trigger requirement
process.triggerStreamResultsFilter = cms.EDFilter('TriggerResultsFilter',
    hltResults = cms.InputTag('TriggerResults', "", "HLT"),
    l1tResults = cms.InputTag(''),                 # L1 uGT results - set to empty to ignore L1
    throw = cms.bool(False),                       # throw exception on unknown trigger names
    triggerConditions = cms.vstring(
        # 13TeV data
        'HLT_HIEle20_WPLoose_Gsf_v*',
        # 13TeV MC
        'HLT_Ele20_WPLoose_Gsf_v*'
        # 5TeV data and MC
        'HLT_HIEle17_WPLoose_Gsf_v*',
        # 5 and 13 TeV data, 5TeV MC
        'HLT_HIMu17_v*',
        # 13TeV MC
        'HLT_Mu17_v*',
    )
)
## count number of events
#process.eventCountPre = cms.EDAnalyzer('EventCounter')
#process.eventCountPost = cms.EDAnalyzer('EventCounter')

# Path and EndPath definitions
if opt.filterTrig:
    #process.nanoAOD_step = cms.Path(process.eventCountPre * process.triggerStreamResultsFilter * process.nanoSequence * process.eventCountPost)
    process.nanoAOD_step = cms.Path(process.triggerStreamResultsFilter * seq)
else:
    #process.nanoAOD_step = cms.Path(process.eventCountPre * process.nanoSequence * process.eventCountPost)
    process.nanoAOD_step = cms.Path(seq)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.NANOAODoutput_step = cms.EndPath(process.NANOAODoutput)

# Schedule definition
process.schedule = cms.Schedule(process.nanoAOD_step,process.endjob_step,process.NANOAODoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(8)
process.options.numberOfStreams=cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)

process.MessageLogger.cerr.FwkReport.reportEvery = 5000
process.MessageLogger.suppressWarning = cms.untracked.vstring('triggerStreamResultsFilter', 'updatedPatJetsWithDeepInfo')

# customisation of the process.

if opt.isMC:
    from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeMC
    process = nanoAOD_customizeMC(process)
else:
    from PhysicsTools.NanoAOD.nano_cff import nanoAOD_customizeData 
    process = nanoAOD_customizeData(process)

# End of customisation functions

# Customisation from command line
#process.TFileService = cms.Service("TFileService", fileName = cms.string(outputname.replace(".root", "_Count.root")) )

process.add_(cms.Service('InitRootHandlers', EnableIMT = cms.untracked.bool(False)))
# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
