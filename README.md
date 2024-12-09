Repo for CS299S Project "Context-aware Token Tree Construction for Speculative Decoding"

The repo is forked from [Sequoia](https://github.com/Infini-AI-Lab/Sequoia).

## Environment Set Up
We recommend the following commands to set up the environment

    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.36.2
    pip install accelerate==0.26.1
    pip install datasets==2.16.1
    pip install einops
    pip install protobuf
    pip install sentencepiece
    pip install typing-extensions

## Evaluations
To reproduce the main results
```
cd tests
bash run_dynamic.sh
```
