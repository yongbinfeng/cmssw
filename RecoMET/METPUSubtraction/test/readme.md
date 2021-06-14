# Instructions to use SONIC producers in CMSSW

## General Information about SONIC in CMSSW

 - [Document](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_230/user-guide/docs/) on the Nvidia Triton Server provided by Nvidia

 - [SONIC Core](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicCore) for the implementation details and explanations of SONICin CMSSW
 - [SONIC Triton](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton) for the implemtation details and explanations of Triton-related SONIC part in CMSSW

Some talks and Updates inside CMS Collaboration:

- [Updates](https://indico.cern.ch/event/1001996/contributions/4288097) in the CMS ML forum on April 7th, 2021
- [Updates](https://indico.cern.ch/event/954163/contributions/4062934) in the CMS ML foruom on October 14th, 2020


## Run SONIC producer examples

Some short toy examples with instructions can be found [here](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton/test) in the test directory under `HeterogeneousCore/SonicTriton`

For the real ML-based algorithm producers deployed in CMSSW, currently we have the DeepMET integrated as the SONIC format. DeepTau is going to be included later. More information about DeepMET algorithm ([twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/DeepPFMET)).

To run the producer, create a clean directory and cd to that directory, then:
```
cmsrel CMSSW_12_0_0_pre1
cd CMSSW_12_0_0_pre1/src
cmsenv
git cms-init
```

Merge the remote branch and compile:
```
git cms-merge-topic yongbinfeng:SonicProduction
scramv1 b -j4
```

Run the test producer:
```
cd RecoMET/METPUSubtraction/test
cmsRun testDeepMETSonic_cfg.py
```

The configuration will run the local fallback sever with Singularity by default, if there is no server available.

## Instructions to make a new SONIC producer

Prepare the CMSSW environment with necessary packages:
```
cmsrel CMSSW_12_0_0_pre1
cd CMSSW_12_0_0_pre1/src
cmsenv
git cms-init
git cms-addpkg HeterogeneousCore/SonicTriton
```

Then add the other packages that are needed for the producer, for example, for DeepMET,
```
git cms-addpkg RecoMET/METPUSubtraction
```

To make a new producer, the producer file, the model config file, the model file, and the python config file are needed.

### Producer

The producer file can be an inherited class based on `TritonEDProducer`, with the `acquire` function to prepare the inputs for the inference server, and `produce` function to receive the outputs from the server. 

More information on the producer [here](https://github.com/cms-sw/cmssw/tree/master/HeterogeneousCore/SonicTriton#modules), and examples: 
- [DeepMET](https://github.com/yongbinfeng/cmssw/blob/SonicProduction/RecoMET/METPUSubtraction/plugins/DeepMETSonicProducer.cc)
- [Toy image producer](https://github.com/cms-sw/cmssw/blob/master/HeterogeneousCore/SonicTriton/test/TritonImageProducer.cc)
- [Toy graph producer](https://github.com/cms-sw/cmssw/blob/master/HeterogeneousCore/SonicTriton/test/TritonGraphModules.cc)


### Model directory
The models (including the config file and the model itself) needs to be placed under `HeterogeneousCore/SonicTriton/data`, following the organization
```
HeterogeneousCore/SonicTriton/data/
  <model-name>/
    config.pbtxt
    1/
      model.pb
```
where `config.pbtxt` is the model configuration, and `1/model.pb` is the model file

#### Model config file

The model config file is needed to define the model name, model format (tensorflow, pytorch, onnx, tensorrt, etc), and the dimension of inputs and the outputs. It needs to be placed under `HeterogeneousCore/SonicTriton/data`, followed by the directory of the model name, with the default name `config.pbtxt`.

More information on the model config file [here](https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton_inference_server_230/user-guide/docs/model_configuration.html), and one example:
- [DeepMET](https://github.com/yongbinfeng/cmssw/blob/SonicProduction/HeterogeneousCore/SonicTriton/data/models/deepmet/config.pbtxt)

#### Model file

The running model can be tensorflow, pytorch, onnx, tensorrt, etc formats.

### Python configuration

A python configuration is needed to set up the server and the client information (model name, mode, path, etc). Two examples:
= DeepMET: [here](https://github.com/yongbinfeng/cmssw/blob/SonicProduction/RecoMET/METPUSubtraction/python/deepMetSonicProducer_cff.py) and [here](https://github.com/yongbinfeng/cmssw/blob/SonicProduction/RecoMET/METPUSubtraction/test/testDeepMETSonic_cfg.py)
- Toy examples: [here](https://github.com/cms-sw/cmssw/blob/master/HeterogeneousCore/SonicTriton/test/tritonTest_cfg.py)
