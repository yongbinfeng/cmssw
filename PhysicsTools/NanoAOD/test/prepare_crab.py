#!/usr/bin/env python
"""
This is a small script that submits a config over datasets
"""
import os

samples = [
    # isData, Dataset name, nfiles_per_job
    (0, '/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2_ext1-v1/MINIAODSIM', 2), # 82 files 
    (0, '/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2_ext1-v1/MINIAODSIM', 2), # 42 files
    (0, '/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2_ext1-v1/MINIAODSIM', 2), # 169
    (0, '/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2_ext1-v1/MINIAODSIM', 2), # 266
    (0, '/WJetsToLNu_0J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2_ext1-v1/MINIAODSIM', 2), # 395
    (0, '/WJetsToLNu_1J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2_ext1-v1/MINIAODSIM', 2), # 79, in production
    (0, '/WJetsToLNu_2J_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2-v1/MINIAODSIM', 2), # 26
    (0, '/WWTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2_ext1-v1/MINIAODSIM', 2), # 6
    (0, '/WZTo3LNu_TuneCP5_13TeV-powheg-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2_ext1-v1/MINIAODSIM', 2), # 7
    (0, '/ZZ_TuneCP5_13TeV-pythia8/RunIIFall17MiniAODv2-fixECALGT_LowPU_94X_mc2017_realistic_v10For2017H_v2_ext1-v1/MINIAODSIM', 2), # 9
    (1, '/HighEGJet/Run2017H-17Nov2017-v1/MINIAOD', 10), # 1070 files in total
    (1, '/SingleMuon/Run2017H-17Nov2017-v2/MINIAOD', 10), # 428 files in total
]

config_dir = 'crab_configs'

if not os.path.isdir( config_dir ) :
    os.mkdir(config_dir )

submit_commands = []

for isData, path, nfiles_per_job in samples:
    if isData:
        pyCfgParams = '["isMC=0", "filterTrig=1"]'
    else: 
        pyCfgParams = '["isMC=1"]'

    base_name = path.split('/')[1]

    fname = '%s/%s.py'%(config_dir, base_name)

    ofile = open(fname, 'w')

    file_entries = []

    file_entries.append('from CRABClient.UserUtilities import config')
    file_entries.append('config = config()')

    file_entries.append('')
    file_entries.append('config.section_("General")')
    file_entries.append('config.General.requestName = "production_LowPUNano_v2_%s"' %(base_name))
    file_entries.append('config.General.workArea = "crab_projects"')
    file_entries.append('config.General.transferLogs = False')
    file_entries.append('config.General.transferOutputs = True')

    file_entries.append('')
    file_entries.append('config.section_("JobType")')
    file_entries.append('config.JobType.pluginName = "Analysis"')
    file_entries.append('config.JobType.psetName = "nano_Run2017.py"')
    file_entries.append('config.JobType.pyCfgParams = %s'%pyCfgParams)
    file_entries.append('config.JobType.maxMemoryMB = 4000 # Default is 2500 : Max I have used is 13000')
    file_entries.append('config.JobType.numCores = 8')
    file_entries.append('config.JobType.allowUndistributedCMSSW = True')

    file_entries.append('')
    file_entries.append('config.section_("Data")')
    file_entries.append('config.Data.inputDataset = "%s"'%path)
    file_entries.append('config.Data.splitting = "FileBased"')
    file_entries.append('config.Data.unitsPerJob = %d'%nfiles_per_job)
    #file_entries.append('config.Data.totalUnits = 100'%nfiles)
    file_entries.append('config.Data.ignoreLocality = False')
    file_entries.append('config.Data.publication = False')
    file_entries.append('config.Data.outputDatasetTag= "NanoAOD_0302"')
    file_entries.append('config.Data.outLFNDirBase = "/store/user/yofeng/LowPU/"')

    file_entries.append('')
    file_entries.append('config.section_("Site")')
    file_entries.append('config.Site.blacklist = ["T2_US_Caltech"]')

    if isData:
        file_entries.append('config.Data.lumiMask = "/afs/cern.ch/work/y/yofeng/public/WpT/CMSSW_10_6_20/src/PhysicsTools/NanoAOD/test/Cert_306896-307082_13TeV_PromptReco_Collisions17_JSON_LowPU_lowPU.txt"')

    file_entries.append('')
    file_entries.append('config.section_("Site")')
    file_entries.append('config.Site.storageSite = "T3_US_FNALLPC"')

    for line in file_entries :
        ofile.write( line + '\n' )

    ofile.close()

    submit_commands.append( 'crab submit --config %s' %( fname ) )


submit_file = open('submit_crab.sh', 'w' )
submit_file.write( '#!/bin/bash\n' )

for cmd in submit_commands :
    submit_file.write(cmd + '\n' )


submit_file.close()
os.system('chmod +x submit_crab.sh')

