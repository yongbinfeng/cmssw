import FWCore.ParameterSet.Config as cms

l1extraParticleMap = cms.EDProducer("L1ExtraParticleMapProd")
l1extraParticleMap.L1_SingleMu14_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleIsoEG8_prescale = cms.int32(1)
l1extraParticleMap.L1_TauJet40_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_ETM10_prescale = cms.int32(1000)
l1extraParticleMap.L1_EG12_ETM30_thresh2 = cms.double(30.0)
l1extraParticleMap.L1_EG12_ETM30_thresh1 = cms.double(12.0)
l1extraParticleMap.L1_HTT100_ETM30_thresh2 = cms.double(30.0)
l1extraParticleMap.L1_Mu3_ETM30_prescale = cms.int32(999999999)
l1extraParticleMap.L1_Mu3_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_Jet70_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_TripleTauJet40_thresh = cms.double(40.0)
l1extraParticleMap.L1_Mu3_Jet70_thresh2 = cms.double(70.0)
l1extraParticleMap.L1_SingleJet70_thresh = cms.double(70.0)
l1extraParticleMap.L1_IsoEG10_TauJet20_prescale = cms.int32(1)
l1extraParticleMap.L1_MinBias_HTT10_prescale = cms.int32(3000000)
l1extraParticleMap.L1_DoubleIsoEG5_HTT200_thresh1 = cms.double(5.0)
l1extraParticleMap.L1_Mu3_Jet70_prescale = cms.int32(999999999)
l1extraParticleMap.L1_IsoEG10_Jet30_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleIsoEG5_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_ETM40_thresh = cms.double(40.0)
l1extraParticleMap.L1_DoubleEG10_ETM20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_ETM15_thresh = cms.double(15.0)
l1extraParticleMap.L1_DoubleTauJet35_thresh = cms.double(35.0)
l1extraParticleMap.L1_IsoEG10_Jet20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_DoubleEG10_ETM20_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_DoubleEG10_Mu3_prescale = cms.int32(999999999)
l1extraParticleMap.L1_ETM50_thresh = cms.double(50.0)
l1extraParticleMap.L1_Mu5_Jet15_prescale = cms.int32(1)
l1extraParticleMap.etMissSource = cms.InputTag("l1extraParticles","MET")
l1extraParticleMap.htMissSource = cms.InputTag("l1extraParticles","MHT")
l1extraParticleMap.L1_TripleEG10_prescale = cms.int32(999999999)
l1extraParticleMap.L1_Mu5_TauJet20_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleMu3_thresh = cms.double(3.0)
l1extraParticleMap.L1_SingleJet200_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleJet50_ETM20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_IsoEG10_Jet20_prescale = cms.int32(1)
l1extraParticleMap.L1_Jet70_ETM40_prescale = cms.int32(999999999)
l1extraParticleMap.L1_DoubleJet70_thresh = cms.double(70.0)
l1extraParticleMap.L1_IsoEG10_Jet70_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleTauJet100_prescale = cms.int32(1)
l1extraParticleMap.L1_HTT300_thresh = cms.double(300.0)
l1extraParticleMap.L1_SingleIsoEG20_thresh = cms.double(20.0)
l1extraParticleMap.L1_DoubleIsoEG10_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleTauJet40_HTT200_thresh1 = cms.double(40.0)
l1extraParticleMap.L1_TripleIsoEG5_thresh = cms.double(5.0)
l1extraParticleMap.nonIsolatedEmSource = cms.InputTag("l1extraParticles","NonIsolated")
l1extraParticleMap.L1_EG10_Jet15_thresh2 = cms.double(15.0)
l1extraParticleMap.L1_SingleTauJet80_thresh = cms.double(80.0)
l1extraParticleMap.L1_SingleMu25_prescale = cms.int32(1)
l1extraParticleMap.L1_Mu5_Jet15_thresh1 = cms.double(5.0)
l1extraParticleMap.L1_SingleJet15_thresh = cms.double(15.0)
l1extraParticleMap.L1_SingleTauJet10_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleMu7_thresh = cms.double(7.0)
l1extraParticleMap.L1_HTT100_ETM30_prescale = cms.int32(999999999)
l1extraParticleMap.L1_TauJet20_ETM20_prescale = cms.int32(999999999)
l1extraParticleMap.L1_Mu5_Jet20_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleTauJet40_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleMu3_EG10_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleIsoEG12_thresh = cms.double(12.0)
l1extraParticleMap.L1_DoubleTauJet40_thresh = cms.double(40.0)
l1extraParticleMap.L1_EG10_Jet15_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_IsoEG10_Jet15_thresh2 = cms.double(15.0)
l1extraParticleMap.muonSource = cms.InputTag("l1extraParticles")
l1extraParticleMap.L1_DoubleIsoEG5_Mu3_thresh2 = cms.double(3.0)
l1extraParticleMap.L1_DoubleIsoEG5_Mu3_thresh1 = cms.double(5.0)
l1extraParticleMap.L1_IsoEG10_Jet15_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_SingleIsoEG25_thresh = cms.double(25.0)
l1extraParticleMap.L1_ETM30_thresh = cms.double(30.0)
l1extraParticleMap.L1_SingleEG10_prescale = cms.int32(100)
l1extraParticleMap.L1_Mu5_TauJet30_prescale = cms.int32(1)
l1extraParticleMap.L1_EG12_Jet70_prescale = cms.int32(999999999)
l1extraParticleMap.L1_ExclusiveDoubleIsoEG4_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleTauJet20_prescale = cms.int32(1000)
l1extraParticleMap.L1_DoubleTauJet35_prescale = cms.int32(999999999)
l1extraParticleMap.L1_IsoEG10_TauJet30_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleEG20_prescale = cms.int32(1)
l1extraParticleMap.L1_IsoEG10_Jet20_ForJet10_thresh3 = cms.double(10.0)
l1extraParticleMap.L1_SingleTauJet40_prescale = cms.int32(1000)
l1extraParticleMap.L1_SingleJet30_thresh = cms.double(30.0)
l1extraParticleMap.L1_IsoEG10_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_IsoEG10_HTT200_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_DoubleMu3_EG10_thresh2 = cms.double(10.0)
l1extraParticleMap.L1_SingleJet15_prescale = cms.int32(100000)
l1extraParticleMap.L1_DoubleMu3_EG10_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_SingleEG25_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleEG10_Mu3_thresh2 = cms.double(3.0)
l1extraParticleMap.L1_DoubleEG10_Mu3_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_Mu3_ETM30_thresh2 = cms.double(30.0)
l1extraParticleMap.L1_Jet70_TauJet40_prescale = cms.int32(999999999)
l1extraParticleMap.L1_Mu3_ETM30_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_ETM40_prescale = cms.int32(1)
l1extraParticleMap.L1_Jet70_ETM40_thresh1 = cms.double(70.0)
l1extraParticleMap.L1_SingleMu7_prescale = cms.int32(1)
l1extraParticleMap.L1_Mu3_Jet15_thresh2 = cms.double(15.0)
l1extraParticleMap.L1_SingleEG10_thresh = cms.double(10.0)
l1extraParticleMap.L1_SingleEG5_prescale = cms.int32(10000)
l1extraParticleMap.L1_EG12_Jet20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_TripleJet50_prescale = cms.int32(1)
l1extraParticleMap.L1_Mu3_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleTauJet10_thresh = cms.double(10.0)
l1extraParticleMap.L1_DoubleEG5_thresh = cms.double(5.0)
l1extraParticleMap.L1_DoubleEG10_thresh = cms.double(10.0)
l1extraParticleMap.L1_Mu5_TauJet20_thresh1 = cms.double(5.0)
l1extraParticleMap.L1_ETM60_thresh = cms.double(60.0)
l1extraParticleMap.L1_HTT200_prescale = cms.int32(100)
l1extraParticleMap.L1_EG12_Jet70_thresh2 = cms.double(70.0)
l1extraParticleMap.L1_ETM30_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleIsoEG5_thresh = cms.double(5.0)
l1extraParticleMap.L1_DoubleIsoEG8_thresh = cms.double(8.0)
l1extraParticleMap.L1_DoubleIsoEG5_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_ETM10_thresh = cms.double(10.0)
l1extraParticleMap.L1_SingleEG5_thresh = cms.double(5.0)
l1extraParticleMap.L1_SingleIsoEG8_thresh = cms.double(8.0)
l1extraParticleMap.L1_IsoEG10_Jet30_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_IsoEG10_TauJet20_thresh2 = cms.double(20.0)
l1extraParticleMap.centralJetSource = cms.InputTag("l1extraParticles","Central")
l1extraParticleMap.L1_Jet70_TauJet40_thresh2 = cms.double(40.0)
l1extraParticleMap.L1_IsoEG10_EG10_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_IsoEG10_EG10_thresh2 = cms.double(10.0)
l1extraParticleMap.L1_Jet70_TauJet40_thresh1 = cms.double(70.0)
l1extraParticleMap.L1_TripleIsoEG5_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleJet50_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleEG12_thresh = cms.double(12.0)
l1extraParticleMap.tauJetSource = cms.InputTag("l1extraParticles","Tau")
l1extraParticleMap.L1_DoubleEG10_prescale = cms.int32(1)
l1extraParticleMap.L1_QuadJet30_prescale = cms.int32(1)
l1extraParticleMap.L1_QuadJet30_thresh = cms.double(30.0)
l1extraParticleMap.L1_IsoEG10_Jet20_ForJet10_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_DoubleTauJet40_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_TauJet30_ETM30_thresh1 = cms.double(30.0)
l1extraParticleMap.L1_TauJet30_ETM30_thresh2 = cms.double(30.0)
l1extraParticleMap.L1_ETM50_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleEG12_prescale = cms.int32(100)
l1extraParticleMap.L1_Mu3_IsoEG5_prescale = cms.int32(1)
l1extraParticleMap.L1_HTT200_thresh = cms.double(200.0)
l1extraParticleMap.L1_SingleMu3_prescale = cms.int32(4000)
l1extraParticleMap.L1_DoubleTauJet30_prescale = cms.int32(100)
l1extraParticleMap.L1_IsoEG10_Jet20_ForJet10_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_HTT100_prescale = cms.int32(1000)
l1extraParticleMap.L1_SingleMu5_thresh = cms.double(5.0)
l1extraParticleMap.L1_SingleJet100_prescale = cms.int32(1)
l1extraParticleMap.L1_TripleEG10_thresh = cms.double(10.0)
l1extraParticleMap.L1_DoubleTauJet40_ETM20_prescale = cms.int32(999999999)
l1extraParticleMap.L1_TripleJet50_thresh = cms.double(50.0)
l1extraParticleMap.L1_TripleMu3_thresh = cms.double(3.0)
l1extraParticleMap.L1_DoubleIsoEG5_Mu3_prescale = cms.int32(999999999)
l1extraParticleMap.L1_DoubleEG15_thresh = cms.double(15.0)
l1extraParticleMap.L1_DoubleJet100_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleEG10_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_SingleJet20_thresh = cms.double(20.0)
l1extraParticleMap.L1_DoubleJet50_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_DoubleEG10_HTT200_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_DoubleTauJet40_ETM20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_DoubleTauJet40_ETM20_thresh1 = cms.double(40.0)
l1extraParticleMap.L1_DoubleJet70_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleEG15_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleTauJet30_thresh = cms.double(30.0)
l1extraParticleMap.L1_EG12_Jet20_prescale = cms.int32(999999999)
l1extraParticleMap.L1_EG12_Jet70_thresh1 = cms.double(12.0)
l1extraParticleMap.L1_SingleTauJet100_thresh = cms.double(100.0)
l1extraParticleMap.L1_SingleMu3_thresh = cms.double(3.0)
l1extraParticleMap.L1_IsoEG10_Jet30_thresh2 = cms.double(30.0)
l1extraParticleMap.L1_SingleTauJet20_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleJet30_prescale = cms.int32(10000)
l1extraParticleMap.L1_Mu5_TauJet20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_HTT250_thresh = cms.double(250.0)
l1extraParticleMap.L1_ETM60_prescale = cms.int32(1)
l1extraParticleMap.L1_ExclusiveJet25_Gap_Jet25_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleMu14_thresh = cms.double(14.0)
l1extraParticleMap.L1_SingleEG15_thresh = cms.double(15.0)
l1extraParticleMap.L1_IsoEG10_EG10_prescale = cms.int32(999999999)
l1extraParticleMap.L1_Jet70_ETM40_thresh2 = cms.double(40.0)
l1extraParticleMap.L1_DoubleMu3_IsoEG5_prescale = cms.int32(999999999)
l1extraParticleMap.L1_IsoEG10_TauJet20_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_ETM15_prescale = cms.int32(300)
l1extraParticleMap.L1_EG12_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_EG12_HTT200_thresh1 = cms.double(12.0)
l1extraParticleMap.L1_SingleMu5_prescale = cms.int32(2000)
l1extraParticleMap.L1_SingleEG20_thresh = cms.double(20.0)
l1extraParticleMap.isolatedEmSource = cms.InputTag("l1extraParticles","Isolated")
l1extraParticleMap.L1_Mu5_TauJet30_thresh2 = cms.double(30.0)
l1extraParticleMap.L1_ExclusiveJet25_Gap_Jet25_thresh = cms.double(25.0)
l1extraParticleMap.L1_ExclusiveDoubleIsoEG4_thresh1 = cms.double(4.0)
l1extraParticleMap.L1_IsoEG10_ETM30_thresh2 = cms.double(30.0)
l1extraParticleMap.L1_SingleEG8_thresh = cms.double(8.0)
l1extraParticleMap.L1_ETM20_prescale = cms.int32(10000)
l1extraParticleMap.L1_DoubleJet50_HTT200_thresh1 = cms.double(50.0)
l1extraParticleMap.L1_HTT400_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleJet20_prescale = cms.int32(999999999)
l1extraParticleMap.L1_ETT60_prescale = cms.int32(1000)
l1extraParticleMap.L1_ExclusiveDoubleIsoEG4_thresh2 = cms.double(10.0)
l1extraParticleMap.L1_TauJet40_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_SingleTauJet40_thresh = cms.double(40.0)
l1extraParticleMap.L1_TauJet20_ETM20_thresh1 = cms.double(20.0)
l1extraParticleMap.L1_TauJet20_ETM20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_TauJet30_ETM40_thresh1 = cms.double(30.0)
l1extraParticleMap.L1_Jet70_HTT200_thresh1 = cms.double(70.0)
l1extraParticleMap.L1_DoubleMu3_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_SingleIsoEG15_thresh = cms.double(15.0)
l1extraParticleMap.L1_ETM20_thresh = cms.double(20.0)
l1extraParticleMap.L1_DoubleJet50_ETM20_prescale = cms.int32(999999999)
l1extraParticleMap.L1_DoubleMu3_ETM20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_SingleEG8_prescale = cms.int32(1000)
l1extraParticleMap.L1_DoubleMu3_ETM20_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_SingleTauJet35_thresh = cms.double(35.0)
l1extraParticleMap.L1_SingleTauJet80_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleEG15_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleTauJet60_prescale = cms.int32(999999999)
l1extraParticleMap.L1_HTT100_thresh = cms.double(100.0)
l1extraParticleMap.L1_SingleTauJet60_thresh = cms.double(60.0)
l1extraParticleMap.L1_HTT300_prescale = cms.int32(1)
l1extraParticleMap.L1_EG12_ETM30_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleJet150_thresh = cms.double(150.0)
l1extraParticleMap.L1_Mu5_IsoEG10_thresh2 = cms.double(10.0)
l1extraParticleMap.L1_SingleMu10_thresh = cms.double(10.0)
l1extraParticleMap.L1_SingleTauJet30_thresh = cms.double(30.0)
l1extraParticleMap.L1_IsoEG10_TauJet30_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_TripleTauJet40_prescale = cms.int32(999999999)
l1extraParticleMap.L1_IsoEG10_TauJet30_thresh2 = cms.double(30.0)
l1extraParticleMap.L1_DoubleIsoEG5_ETM20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_HTT500_thresh = cms.double(500.0)
l1extraParticleMap.L1_ETT60_thresh = cms.double(60.0)
l1extraParticleMap.L1_Mu3_IsoEG5_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_ExclusiveDoubleJet60_thresh = cms.double(60.0)
l1extraParticleMap.L1_TauJet30_ETM30_prescale = cms.int32(1)
l1extraParticleMap.L1_EG12_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_DoubleJet50_ETM20_thresh1 = cms.double(50.0)
l1extraParticleMap.L1_Mu5_IsoEG10_thresh1 = cms.double(5.0)
l1extraParticleMap.L1_DoubleTauJet40_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_TripleMu3_prescale = cms.int32(1)
l1extraParticleMap.L1_IsoEG10_ETM30_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_Mu3_EG12_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_Mu5_TauJet30_thresh1 = cms.double(5.0)
l1extraParticleMap.L1_SingleJet50_thresh = cms.double(50.0)
l1extraParticleMap.L1_Mu3_EG12_thresh2 = cms.double(12.0)
l1extraParticleMap.L1_DoubleJet100_thresh = cms.double(100.0)
l1extraParticleMap.L1_HTT500_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleIsoEG15_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleIsoEG5_ETM20_thresh1 = cms.double(5.0)
l1extraParticleMap.L1_Mu3_HTT200_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_EG10_Jet15_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleIsoEG12_prescale = cms.int32(1)
l1extraParticleMap.L1_EG12_TauJet40_thresh1 = cms.double(12.0)
l1extraParticleMap.L1_SingleMu25_thresh = cms.double(25.0)
l1extraParticleMap.L1_IsoEG10_Jet20_ForJet10_prescale = cms.int32(1)
l1extraParticleMap.L1_HTT250_prescale = cms.int32(1)
l1extraParticleMap.L1_Mu3_IsoEG5_thresh2 = cms.double(5.0)
l1extraParticleMap.L1_DoubleEG10_ETM20_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleIsoEG10_thresh = cms.double(10.0)
l1extraParticleMap.L1_SingleTauJet35_prescale = cms.int32(999999999)
l1extraParticleMap.L1_EG12_TauJet40_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleIsoEG10_prescale = cms.int32(100)
l1extraParticleMap.L1_HTT100_ETM30_thresh1 = cms.double(100.0)
l1extraParticleMap.L1_SingleJet70_prescale = cms.int32(100)
l1extraParticleMap.L1_SingleJet200_thresh = cms.double(200.0)
l1extraParticleMap.L1_Mu5_Jet15_thresh2 = cms.double(15.0)
l1extraParticleMap.L1_Mu3_Jet70_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_SingleJet150_prescale = cms.int32(1)
l1extraParticleMap.L1_IsoEG10_Jet70_thresh2 = cms.double(70.0)
l1extraParticleMap.L1_IsoEG10_Jet70_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_TauJet30_ETM40_thresh2 = cms.double(40.0)
l1extraParticleMap.L1_DoubleEG5_prescale = cms.int32(10000)
l1extraParticleMap.L1_SingleMu20_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleTauJet20_thresh = cms.double(20.0)
l1extraParticleMap.L1_Mu3_Jet15_prescale = cms.int32(20)
l1extraParticleMap.L1_SingleJet100_thresh = cms.double(100.0)
l1extraParticleMap.L1_EG12_TauJet40_thresh2 = cms.double(40.0)
l1extraParticleMap.L1_Jet70_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_SingleIsoEG5_prescale = cms.int32(10000)
l1extraParticleMap.L1_DoubleIsoEG10_thresh = cms.double(10.0)
l1extraParticleMap.L1_Mu3_Jet15_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_DoubleMu3_prescale = cms.int32(1)
l1extraParticleMap.L1_SingleIsoEG20_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleMu3_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleEG25_thresh = cms.double(25.0)
l1extraParticleMap.forwardJetSource = cms.InputTag("l1extraParticles","Forward")
l1extraParticleMap.L1_DoubleJet50_HTT200_thresh2 = cms.double(200.0)
l1extraParticleMap.L1_DoubleIsoEG5_ETM20_prescale = cms.int32(999999999)
l1extraParticleMap.L1_DoubleMu3_IsoEG5_thresh2 = cms.double(5.0)
l1extraParticleMap.L1_IsoEG10_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_SingleTauJet30_prescale = cms.int32(999999999)
l1extraParticleMap.L1_Mu5_IsoEG10_prescale = cms.int32(1)
l1extraParticleMap.L1_ZeroBias_prescale = cms.int32(3000000)
l1extraParticleMap.L1_SingleIsoEG25_prescale = cms.int32(1)
l1extraParticleMap.L1_DoubleEG10_HTT200_prescale = cms.int32(999999999)
l1extraParticleMap.L1_DoubleMu3_ETM20_prescale = cms.int32(999999999)
l1extraParticleMap.L1_ExclusiveDoubleJet60_prescale = cms.int32(1)
l1extraParticleMap.L1_IsoEG10_Jet20_thresh1 = cms.double(10.0)
l1extraParticleMap.L1_Mu5_Jet20_thresh2 = cms.double(20.0)
l1extraParticleMap.L1_SingleTauJet20_thresh = cms.double(20.0)
l1extraParticleMap.L1_Mu5_Jet20_thresh1 = cms.double(5.0)
l1extraParticleMap.L1_SingleMu20_thresh = cms.double(20.0)
l1extraParticleMap.L1_HTT400_thresh = cms.double(400.0)
l1extraParticleMap.L1_SingleIsoEG8_prescale = cms.int32(1000)
l1extraParticleMap.L1_TauJet40_HTT200_thresh1 = cms.double(40.0)
l1extraParticleMap.L1_DoubleMu3_HTT200_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_IsoEG10_Jet15_prescale = cms.int32(20)
l1extraParticleMap.L1_IsoEG10_ETM30_prescale = cms.int32(999999999)
l1extraParticleMap.L1_DoubleMu3_IsoEG5_thresh1 = cms.double(3.0)
l1extraParticleMap.L1_TauJet30_ETM40_prescale = cms.int32(1)
l1extraParticleMap.L1_EG12_Jet20_thresh1 = cms.double(12.0)
l1extraParticleMap.L1_SingleMu10_prescale = cms.int32(1)
l1extraParticleMap.L1_Mu3_EG12_prescale = cms.int32(1)


