## Instructions to use SONIC producers in CMSSW

### General Information about SONIC in CMSSW

 - [Document](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_230/user-guide/docs/) on the Nvidia Triton Server provided by Nvidia

 - [SONIC Core](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicCore) for the implementation details and explanations of SONICin CMSSW
 - [SONIC Triton](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton) for the implemtation details and explanations of Triton-related SONIC part in CMSSW

Some talks and Updates inside CMS Collaboration:

- [Updates](https://indico.cern.ch/event/1001996/contributions/4288097) in the CMS ML forum on April 7th, 2021
- [Updates](https://indico.cern.ch/event/954163/contributions/4062934) in the CMS ML foruom on October 14th, 2020


### Run SONIC producer examples

Some short toy examples with instructions can be found [here](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton/test) in the test directory under `HeterogeneousCore/SonicTriton`

Currently for the ML-based algorithm producers deployed in CMSSW, we have the DeepMET integrated as the SONIC format. More information about DeepMET algorithm ([twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/DeepPFMET)).

To run the producer, create a clean directory and cd to that directory, then

```
cmsrel CMSSW_12_0_0_pre1
cd CMSSW_12_0_0_pre1/src
cmsenv
git cms-init
```

Merge the remote branch and compile
```
git cms-merge-topic yongbinfeng:SonicProduction
scramv1 b -j4
```

Run the test producer
```
cd RecoMET/METPUSubtraction/test
cmsRun testDeepMETSonic_cfg.py
```

The configuration will run the local fallback sever with Singularity by default, if there is no server available.

### Instructions to maek a SONIC producer

```
cmsrel CMSSW_12_0_0_pre1
cd CMSSW_12_0_0_pre1/src
cmsenv
git cms-init
git cms-addpkg HeterogeneousCore/SonicTriton
```

Then add the other packages that are needed for the producer, for example, for DeepMET, add
```
git cms-addpkg RecoMET/METPUSubtraction
```

Compile the packages locally by
```
scramv1 b -j4
```


