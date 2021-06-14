## Instructions to use SONIC producers in CMSSW

### General Information about SONIC in CMSSW

Document on the Nvidia Triton Server provided by Nvidia: https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_230/user-guide/docs/quickstart.html

Useful explanations for the structure and functions with SONIC:

SONIC Core: https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicCore
SONIC Triton: https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton

Talks and Updates inside CMS Collaboration:

- Updates in the CMS ML forum on April 7th, 2021: https://indico.cern.ch/event/1001996/contributions/4288097
- updates in the CMS ML foruom on October 14th, 2020: https://indico.cern.ch/event/954163/contributions/4062934


### Real SONIC producer examples

Currently for the ML-based algorithm producers deployed in CMSSW, we have the DeepMET integrated as the SONIC format. More information about DeepMET algorithm ([twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/DeepPFMET)).

To run the producer, create a clean directory and cd to that directory, then

```
cmsrel CMSSW_12_0_0_pre1
cd CMSSW_12_0_0_pre1/src
cmsenv
git cms-init
```

Merge the remote branch
```
git cms-merge-topic yongbinfeng:SonicProduction
```


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


